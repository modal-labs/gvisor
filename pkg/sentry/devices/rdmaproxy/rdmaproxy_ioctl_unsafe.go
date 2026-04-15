// Copyright 2024 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rdmaproxy

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/cleanup"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/fdnotifier"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/sentry/arch"
	"gvisor.dev/gvisor/pkg/sentry/kernel"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
	"gvisor.dev/gvisor/pkg/sentry/mm"
	"gvisor.dev/gvisor/pkg/sentry/vfs"
	"gvisor.dev/gvisor/pkg/usermem"
	"gvisor.dev/gvisor/pkg/waiter"
)

// ioctlInHostNetns executes an ioctl in the host's network namespace.
// RoCE's ibv_modify_qp requires the calling thread's network namespace
// to contain the physical NICs referenced by GIDs. Both FDs are saved
// at startup (before seccomp) so no open() is needed here.
func ioctlInHostNetns(fd int32, cmd uint32, arg unsafe.Pointer) (uintptr, unix.Errno) {
	if hostNetnsFD < 0 || containerNetnsFD < 0 {
		n, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd), uintptr(cmd), uintptr(arg))
		return n, errno
	}

	if netnsFDsSameInode(hostNetnsFD, containerNetnsFD) {
		n, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd), uintptr(cmd), uintptr(arg))
		return n, errno
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := unix.Setns(int(hostNetnsFD), unix.CLONE_NEWNET); err != nil {
		log.Warningf("rdmaproxy: setns to host netns: %v, falling back", err)
		n, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd), uintptr(cmd), uintptr(arg))
		return n, errno
	}

	n, _, errno := unix.RawSyscall(unix.SYS_IOCTL, uintptr(fd), uintptr(cmd), uintptr(arg))

	if restoreErr := unix.Setns(int(containerNetnsFD), unix.CLONE_NEWNET); restoreErr != nil {
		log.Warningf("rdmaproxy: restore container netns: %v", restoreErr)
	}

	return n, errno
}

// ioctlDirect executes an ioctl without network namespace switching.
// Used for operations that don't need the host netns (MR, CQ, QP ops).
func ioctlDirect(fd int32, cmd uint32, arg unsafe.Pointer) (uintptr, unix.Errno) {
	n, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(fd), uintptr(cmd), uintptr(arg))
	return n, errno
}

// writeInHostNetns runs write(2) on the uverbs fd in the host network namespace.
// Legacy ibv_modify_qp uses the write() command path (IB_USER_VERBS_CMD_MODIFY_QP);
// it needs the same netns as ioctl MODIFY_QP for RoCE GID resolution.
func writeInHostNetns(fd int32, p []byte) (uintptr, unix.Errno) {
	if len(p) == 0 {
		return 0, 0
	}
	if hostNetnsFD < 0 || containerNetnsFD < 0 {
		n, _, errno := unix.RawSyscall(unix.SYS_WRITE,
			uintptr(fd),
			uintptr(unsafe.Pointer(&p[0])),
			uintptr(len(p)))
		return n, errno
	}

	if netnsFDsSameInode(hostNetnsFD, containerNetnsFD) {
		n, _, errno := unix.RawSyscall(unix.SYS_WRITE,
			uintptr(fd),
			uintptr(unsafe.Pointer(&p[0])),
			uintptr(len(p)))
		return n, errno
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := unix.Setns(int(hostNetnsFD), unix.CLONE_NEWNET); err != nil {
		log.Warningf("rdmaproxy: setns to host netns (write): %v, falling back", err)
		n, _, errno := unix.RawSyscall(unix.SYS_WRITE,
			uintptr(fd),
			uintptr(unsafe.Pointer(&p[0])),
			uintptr(len(p)))
		return n, errno
	}

	n, _, errno := unix.RawSyscall(unix.SYS_WRITE,
		uintptr(fd),
		uintptr(unsafe.Pointer(&p[0])),
		uintptr(len(p)))

	if restoreErr := unix.Setns(int(containerNetnsFD), unix.CLONE_NEWNET); restoreErr != nil {
		log.Warningf("rdmaproxy: restore container netns (write): %v", restoreErr)
	}

	return n, errno
}

// Legacy write / INVOKE_WRITE command for MODIFY_QP (enum ib_uverbs_write_cmds).
// Needs host netns for RoCE GID resolution during QP state transitions.
const ibUserVerbsCmdModifyQP = 26

// uverbsWriteCmdBase strips IB_USER_VERBS_CMD_FLAG_EXTENDED (0x80000000) from
// UVERBS_ATTR_WRITE_CMD values (see ib_user_verbs.h).
func uverbsWriteCmdBase(w uint64) uint32 {
	return uint32(w & 0x7fffffff)
}

// netnsFDsSameInode returns true if two /proc/.../ns/net fds refer to the same
// namespace (same inode). Used to skip setns when the sandbox already shares
// the host network namespace (e.g. Docker --network=host).
func netnsFDsSameInode(a, b int32) bool {
	if a < 0 || b < 0 {
		return false
	}
	var sa, sb unix.Stat_t
	if unix.Fstat(int(a), &sa) != nil {
		return false
	}
	if unix.Fstat(int(b), &sb) != nil {
		return false
	}
	return sa.Ino == sb.Ino && sa.Dev == sb.Dev
}

// ioctlNeedsHostNetns returns true if the ioctl requires the host
// network namespace. Only QP MODIFY operations need it (for GID
// resolution on RoCE). Everything else — QUERY_GID_ENTRY, MR ops,
// CQ ops, QP CREATE/DESTROY — does not.
func ioctlNeedsHostNetns(action ioctlAction, objectID uint16, writeCmdVal uint64) bool {
	// Known safe actions never need netns.
	switch action {
	case actionMRReg, actionMRRegDMABuf, actionMRDereg,
		actionCQCreate, actionCQDestroy,
		actionQPCreate, actionQPDestroy:
		return false
	}
	// INVOKE_WRITE with MODIFY_QP needs netns.
	if objectID == uverbsObjectDevice && uverbsWriteCmdBase(writeCmdVal) == ibUserVerbsCmdModifyQP {
		return true
	}
	// QP object operations other than create/destroy (i.e. MODIFY) need netns.
	if objectID == uverbsObjectQP {
		return true
	}
	// Everything else (QUERY_GID_ENTRY, QUERY_PORT, ALLOC_PD, etc.) doesn't.
	return false
}

// Performance counters for diagnosing throughput regressions.
var (
	globalIoctlCount     atomic.Uint64
	globalWriteCount     atomic.Uint64
	globalReadCount      atomic.Uint64
	globalMRRegCount     atomic.Uint64
	globalMRDeregCount   atomic.Uint64
	globalCQCreateCount  atomic.Uint64
	globalCQDestroyCount atomic.Uint64
	globalQPCreateCount  atomic.Uint64
	globalQPDestroyCount atomic.Uint64
	reporterOnce         sync.Once
)

func startPerfReporter() {
	reporterOnce.Do(func() {
		go func() {
			var prevIoctl, prevWrite, prevRead uint64
			var prevMRReg, prevMRDereg uint64
			var prevCQCreate, prevCQDestroy uint64
			var prevQPCreate, prevQPDestroy uint64
			for {
				time.Sleep(5 * time.Second)
				curIoctl := globalIoctlCount.Load()
				curWrite := globalWriteCount.Load()
				curRead := globalReadCount.Load()
				curMRReg := globalMRRegCount.Load()
				curMRDereg := globalMRDeregCount.Load()
				curCQCreate := globalCQCreateCount.Load()
				curCQDestroy := globalCQDestroyCount.Load()
				curQPCreate := globalQPCreateCount.Load()
				curQPDestroy := globalQPDestroyCount.Load()
				log.Warningf("rdmaproxy: PERF ioctl_rate=%d/5s write_rate=%d/5s read_rate=%d/5s mr_reg=%d mr_dereg=%d cq_create=%d cq_destroy=%d qp_create=%d qp_destroy=%d total_ioctls=%d",
					curIoctl-prevIoctl, curWrite-prevWrite, curRead-prevRead,
					curMRReg-prevMRReg, curMRDereg-prevMRDereg,
					curCQCreate-prevCQCreate, curCQDestroy-prevCQDestroy,
					curQPCreate-prevQPCreate, curQPDestroy-prevQPDestroy,
					curIoctl)
				prevIoctl, prevWrite, prevRead = curIoctl, curWrite, curRead
				prevMRReg, prevMRDereg = curMRReg, curMRDereg
				prevCQCreate, prevCQDestroy = curCQCreate, curCQDestroy
				prevQPCreate, prevQPDestroy = curQPCreate, curQPDestroy
			}
		}()
	})
}

// ioctlBufPool reuses ioctl buffers to avoid per-call heap allocations.
var ioctlBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, hostarch.PageSize)
		return &b
	},
}

// ib_uverbs_ioctl_hdr layout constants.
const (
	rdmaIoctlMagic       = 0x1b
	ibUverbsIoctlHdrSize = 24
	ibUverbsAttrSize     = 16
)

// UVERBS object types (from include/uapi/rdma/ib_user_ioctl_cmds.h).
const (
	uverbsObjectDevice     = 0
	uverbsObjectCQ         = 3
	uverbsObjectQP         = 4
	uverbsObjectMR         = 7
	uverbsObjectAsyncEvent = 16
)

// UVERBS_OBJECT_ASYNC_EVENT method and attr IDs.
const (
	uverbsMethodAsyncEventAlloc = 0
	uverbsAttrAsyncEventAllocFD = 0
)

// CQ CREATE attr IDs (from include/uapi/rdma/ib_user_ioctl_cmds.h).
const (
	uverbsAttrCreateCQEventFD = 7 // UVERBS_ATTR_CREATE_CQ_EVENT_FD
)

// UVERBS method IDs.
const (
	uverbsMethodInvokeWrite = 0  // DEVICE object
	uverbsMethodMRDestroy   = 1  // MR object
	uverbsMethodRegDMABufMR = 4  // MR object (DMABUF path)
	uverbsMethodRegMR       = 5  // MR object (modern path)
	uverbsMethodCoreCreate  = 64 // UVERBS_API_METHOD_KEY_NUM_CORE — CREATE for CQ, QP, etc.
)

// INVOKE_WRITE attr IDs.
const (
	uverbsAttrCoreIn   = 0
	uverbsAttrCoreOut  = 1
	uverbsAttrWriteCmd = 2
)

// Legacy write command numbers (from include/uapi/rdma/ib_user_verbs.h).
const (
	ibUserVerbsCmdCreateCQ  = 6
	ibUserVerbsCmdCreateQP  = 8
	ibUserVerbsCmdRegMR     = 9
	ibUserVerbsCmdDestroyCQ = 11
	ibUserVerbsCmdDeregMR   = 13
	ibUserVerbsCmdDestroyQP = 14
)

// mlx5 driver attr offsets for CQ/QP CREATE (struct mlx5_ib_create_cq / mlx5_ib_create_qp).
const (
	driverAttrBufAddr = 0  // __aligned_u64 buf_addr
	driverAttrDBAddr  = 8  // __aligned_u64 db_addr
	driverAttrMinLen  = 16 // minimum driver attr size we need
)

// mlx5 driver attr IDs.
const (
	mlx5DriverAttrIn  = 0x1000 // input driver data
	mlx5DriverAttrOut = 0x1001 // output driver data
)

// ioctlAction classifies what an ioctl does for page-mirroring purposes.
type ioctlAction int

const (
	actionNone ioctlAction = iota
	actionMRReg
	actionMRRegDMABuf
	actionMRDereg
	actionCQCreate
	actionCQDestroy
	actionQPCreate
	actionQPDestroy
)

func countAction(action ioctlAction) {
	switch action {
	case actionMRReg, actionMRRegDMABuf:
		globalMRRegCount.Add(1)
	case actionMRDereg:
		globalMRDeregCount.Add(1)
	case actionCQCreate:
		globalCQCreateCount.Add(1)
	case actionCQDestroy:
		globalCQDestroyCount.Add(1)
	case actionQPCreate:
		globalQPCreateCount.Add(1)
	case actionQPDestroy:
		globalQPDestroyCount.Add(1)
	}
}

func actionFromLegacyWriteCmd(cmdBase uint32) ioctlAction {
	switch cmdBase {
	case ibUserVerbsCmdRegMR:
		return actionMRReg
	case ibUserVerbsCmdDeregMR:
		return actionMRDereg
	case ibUserVerbsCmdCreateCQ:
		return actionCQCreate
	case ibUserVerbsCmdDestroyCQ:
		return actionCQDestroy
	case ibUserVerbsCmdCreateQP:
		return actionQPCreate
	case ibUserVerbsCmdDestroyQP:
		return actionQPDestroy
	default:
		return actionNone
	}
}

func taskLogFields(t *kernel.Task) string {
	if t == nil {
		return "tid=0 tgid_root=0"
	}
	return fmt.Sprintf("tid=%d tgid_root=%d", t.ThreadID(), t.TGIDInRoot())
}

func formatMRSummary(t *kernel.Task, sandboxVA, length uint64, sentryVA uintptr, oldHCAVA, newHCAVA, oldIOVA, newIOVA uint64) string {
	relocated := sentryVA != uintptr(sandboxVA)
	return fmt.Sprintf("app=%#x-%#x len=%d sentry=%#x-%#x relocated=%t hca_va=%#x->%#x iova=%#x->%#x %s",
		sandboxVA, sandboxVA+length, length,
		sentryVA, sentryVA+uintptr(length), relocated,
		oldHCAVA, newHCAVA, oldIOVA, newIOVA, taskLogFields(t))
}

// ib_uverbs_reg_mr struct field offsets.
const (
	regMROffStart  = 8  // __aligned_u64 start
	regMROffLength = 16 // __aligned_u64 length
	regMROffHcaVA  = 24 // __aligned_u64 hca_va
)

// ib_uverbs_reg_mr_resp field offsets.
const (
	regMRRespOffHandle = 0 // __u32 mr_handle
)

// ib_uverbs_dereg_mr field offsets.
const (
	deregMROffHandle = 0 // __u32 mr_handle
)

// REG_MR attr IDs (modern path).
const (
	uverbsAttrRegMRHandle = 0
	uverbsAttrRegMRIova   = 3
	uverbsAttrRegMRAddr   = 4
	uverbsAttrRegMRLength = 5
)


// DESTROY_MR attr IDs.
const (
	uverbsAttrDestroyMRHandle = 0

	// UVERBS_METHOD_REG_DMABUF_MR attr IDs (enum uverbs_attrs_reg_dmabuf_mr_cmd_attr_ids).
	uverbsAttrRegDMABufMRHandle = 0 // Output: MR handle
	uverbsAttrRegDMABufMRFD     = 5 // Input: DMA-BUF file descriptor (inline)
)

// RDMA_VERBS_IOCTL = _IOWR(0x1b, 1, struct ib_uverbs_ioctl_hdr)
var rdmaVerbsIoctl = linux.IOWR(rdmaIoctlMagic, 1, ibUverbsIoctlHdrSize)

// attrRewrite tracks a sandbox pointer that was rewritten to a sentry buffer.
type attrRewrite struct {
	attrOff  int
	origData uint64
	sentry   []byte
}

// Ioctl implements vfs.FileDescriptionImpl.Ioctl.
func (fd *uverbsFD) Ioctl(ctx context.Context, uio usermem.IO, sysno uintptr, args arch.SyscallArguments) (uintptr, error) {
	cmd := args[1].Uint()
	argPtr := args[2].Pointer()

	t := kernel.TaskFromContext(ctx)
	if t == nil {
		log.Warningf("rdmaproxy: ioctl called without task context")
		return 0, unix.EINVAL
	}

	if cmd == rdmaVerbsIoctl {
		return fd.handleRDMAVerbsIoctl(t, argPtr)
	}
	log.Warningf("rdmaproxy: unhandled ioctl cmd=0x%x (magic=0x%x nr=%d size=%d) on hostFD=%d",
		cmd, (cmd>>8)&0xff, cmd&0xff, (cmd>>16)&0x3fff, fd.hostFD)
	return 0, linuxerr.ENOSYS
}

// handleRDMAVerbsIoctl handles the modern RDMA_VERBS_IOCTL which uses a
// self-describing header + variable-length attribute array.
func (fd *uverbsFD) handleRDMAVerbsIoctl(t *kernel.Task, argPtr hostarch.Addr) (uintptr, error) {
	globalIoctlCount.Add(1)
	startPerfReporter()

	// Single CopyIn: read first 8 bytes to get length, then full buffer.
	var lenBuf [8]byte
	if _, err := t.CopyInBytes(argPtr, lenBuf[:]); err != nil {
		log.Warningf("rdmaproxy: ioctl CopyIn length from %#x: %v", argPtr, err)
		return 0, err
	}
	length := binary.LittleEndian.Uint16(lenBuf[0:2])
	numAttrs := binary.LittleEndian.Uint16(lenBuf[6:8])

	expectedLen := uint16(ibUverbsIoctlHdrSize) + numAttrs*uint16(ibUverbsAttrSize)
	if length != expectedLen || length > hostarch.PageSize {
		log.Warningf("rdmaproxy: ioctl bad header: length=%d expected=%d (numAttrs=%d)",
			length, expectedLen, numAttrs)
		return 0, linuxerr.EINVAL
	}

	// Get buffer from pool to avoid per-ioctl allocation.
	bufPtr := ioctlBufPool.Get().(*[]byte)
	buf := (*bufPtr)[:length]
	defer func() { ioctlBufPool.Put(bufPtr) }()

	if _, err := t.CopyInBytes(argPtr, buf); err != nil {
		log.Warningf("rdmaproxy: ioctl CopyIn full buffer (%d bytes) from %#x: %v",
			length, argPtr, err)
		return 0, err
	}

	objectID := binary.LittleEndian.Uint16(buf[2:4])
	methodID := binary.LittleEndian.Uint16(buf[4:6])
	reserved1 := binary.LittleEndian.Uint64(buf[8:16])
	driverID := binary.LittleEndian.Uint32(buf[16:20])

	log.Debugf("rdmaproxy: IOCTL hostFD=%d obj=0x%04x method=%d attrs=%d len=%d reserved=%#x driver=%d",
		fd.hostFD, objectID, methodID, numAttrs, length, reserved1, driverID)

	// Walk attrs: probe each data field to determine if it's a sandbox
	// pointer (CopyIn succeeds) or inline data (CopyIn fails).
	// Pre-allocate rewrites on stack for the common case.
	var rewritesBuf [16]attrRewrite
	rewrites := rewritesBuf[:0]
	// Stack arena for attribute data to avoid per-attr heap allocation.
	var attrArena [4096]byte
	arenaOff := 0

	for i := 0; i < int(numAttrs); i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		attrLen := binary.LittleEndian.Uint16(buf[off+2 : off+4])
		attrFlags := binary.LittleEndian.Uint16(buf[off+4 : off+6])
		attrData := binary.LittleEndian.Uint64(buf[off+8 : off+16])

		if attrLen == 0 {
			log.Debugf("rdmaproxy:   attr[%d] id=0x%04x len=0 flags=0x%04x data=%#016x (handle/fd)",
				i, attrID, attrFlags, attrData)
			continue
		}

		// Use stack arena when possible, fall back to heap for large attrs.
		var sb []byte
		if arenaOff+int(attrLen) <= len(attrArena) {
			sb = attrArena[arenaOff : arenaOff+int(attrLen)]
			arenaOff += int(attrLen)
		} else {
			sb = make([]byte, attrLen)
		}
		_, copyErr := t.CopyInBytes(hostarch.Addr(attrData), sb)
		if copyErr == nil {
			log.Debugf("rdmaproxy:   attr[%d] id=0x%04x len=%d flags=0x%04x data=ptr:%#016x (rewrite)",
				i, attrID, attrLen, attrFlags, attrData)
			if attrLen <= 64 {
				log.Debugf("rdmaproxy:   attr[%d] data: %x", i, sb)
			} else {
				log.Debugf("rdmaproxy:   attr[%d] data (first 64): %x ...", i, sb[:64])
			}
			binary.LittleEndian.PutUint64(buf[off+8:off+16],
				uint64(uintptr(unsafe.Pointer(&sb[0]))))
			rewrites = append(rewrites, attrRewrite{
				attrOff:  off,
				origData: attrData,
				sentry:   sb,
			})
		} else {
			log.Debugf("rdmaproxy:   attr[%d] id=0x%04x len=%d flags=0x%04x data=inline:%#016x",
				i, attrID, attrLen, attrFlags, attrData)
		}
	}

	// Rewrite inline FD attrs that reference proxied async event FDs.
	// The application sees sentry FD numbers, but the host kernel needs
	// the original host FDs (e.g. CQ CREATE's comp channel attr).
	//
	// IMPORTANT: Only rewrite attrs with known FD attr IDs. Other inline
	// attrs carry kernel object handles (PD, CQ, QP handles) that have
	// small numeric values which could collide with sandbox FD numbers.
	// Rewriting those would corrupt the handles (e.g., PD handle 92 →
	// host FD 3480 → ibv_create_qp ENOENT).
	const (
		uverbsAttrCQCompChannel = 0x0007 // CQ CREATE comp channel FD
		uverbsAttrQPEventFD     = 0x000c // QP CREATE event FD
	)
	for i := 0; i < int(numAttrs); i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		attrLen := binary.LittleEndian.Uint16(buf[off+2 : off+4])
		if attrLen != 0 {
			continue
		}
		// Only rewrite known FD attrs, not handles.
		if attrID != uverbsAttrCQCompChannel && attrID != uverbsAttrQPEventFD {
			continue
		}
		sentryVal := int32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		if sentryVal <= 0 {
			continue
		}
		file, _ := t.FDTable().Get(sentryVal)
		if file == nil {
			continue
		}
		if afd, ok := file.Impl().(*asyncEventFD); ok {
			binary.LittleEndian.PutUint64(buf[off+8:off+16], uint64(afd.hostFD))
			log.Debugf("rdmaproxy: REWRITE obj=0x%04x method=%d attr[%d] id=0x%04x sentry=%d → host=%d (async event FD)", objectID, methodID, i, attrID, sentryVal, afd.hostFD)
		}
		file.DecRef(t)
	}

	// DMABUF MR_REG (method 4): translate the DMA-BUF fd from the
	// sandbox's fd table to the host fd. The DMA-BUF fd is attr 5
	// (UVERBS_ATTR_REG_DMABUF_MR_FD), stored inline.
	if objectID == uverbsObjectMR && methodID == uverbsMethodRegDMABufMR {
		fd.rewriteDMABufFD(t, buf, int(numAttrs))
	}

	// Classify and prepare DMA page mirroring before forwarding.
	action, writeCmdVal := fd.classifyIoctl(buf, int(numAttrs), objectID, methodID)
	countAction(action)

	var mrMirror *mirroredPages
	var cqqpMirror *pinnedDMABufs
	var dmaCleanup cleanup.Cleanup
	defer dmaCleanup.Clean()

	switch action {
	case actionMRReg:
		var err error
		mrMirror, err = fd.prepareMRReg(t, buf, int(numAttrs), objectID, rewrites, writeCmdVal)
		if err != nil {
			log.Warningf("rdmaproxy: MR REG page mirroring: %v", err)
			return 0, linuxerr.ENOMEM
		}
		if mrMirror != nil {
			dmaCleanup = cleanup.Make(func() { mrMirror.release(t) })
		}

	case actionMRRegDMABuf:
		// DMA-BUF MR: no page mirroring needed. The kernel resolves
		// GPU pages through the DMA-BUF framework, not through VMAs.
		log.Debugf("rdmaproxy: DMABUF MR_REG on hostFD=%d — forwarding without page mirroring", fd.hostFD)

	case actionCQCreate, actionQPCreate:
		var err error
		cqqpMirror, err = fd.prepareCQQPCreate(t, buf, int(numAttrs), rewrites, action)
		if err != nil {
			log.Warningf("rdmaproxy: CQ/QP CREATE page mirroring: %v", err)
			return 0, linuxerr.ENOMEM
		}
		if cqqpMirror != nil {
			dmaCleanup = cleanup.Make(func() { cqqpMirror.release(t) })
		}
	}

	log.Debugf("rdmaproxy: forwarding ioctl to host (hostFD=%d, %d rewrites, action=%d)", fd.hostFD, len(rewrites), action)

	var n uintptr
	var errno unix.Errno
	if ioctlNeedsHostNetns(action, objectID, writeCmdVal) {
		n, errno = ioctlInHostNetns(fd.hostFD, rdmaVerbsIoctl, unsafe.Pointer(&buf[0]))
	} else {
		n, errno = ioctlDirect(fd.hostFD, rdmaVerbsIoctl, unsafe.Pointer(&buf[0]))
	}

	if errno != 0 {
		log.Debugf("rdmaproxy: host ioctl returned n=%d errno=%d (%v)", n, errno, errno)
		if errno == unix.EFAULT {
			// Extract the sentry VA that was passed to the host from the ioctl buffer.
			var sentryVA, mrLen uint64
			if action == actionMRReg {
				if rw := findRewrite(buf, int(numAttrs), rewrites, uverbsAttrCoreIn); rw != nil && len(rw.sentry) >= regMROffLength+8 {
					sentryVA = binary.LittleEndian.Uint64(rw.sentry[regMROffStart : regMROffStart+8])
					mrLen = binary.LittleEndian.Uint64(rw.sentry[regMROffLength : regMROffLength+8])
				}
			}
			gpuCached := mrMirror != nil && mrMirror.gpuVMACached
			tgid := int32(t.TGIDInRoot())
			log.Warningf("rdmaproxy: EFAULT from host ioctl obj=0x%04x method=%d action=%d hostFD=%d sentryVA=%#x mrLen=%d gpuCached=%v tgid=%d (%s)",
				objectID, methodID, action, fd.hostFD, sentryVA, mrLen, gpuCached, tgid, taskLogFields(t))
		}
	} else {
		log.Debugf("rdmaproxy: host ioctl returned n=%d OK", n)
	}

	// Post-ioctl tracking for successful operations.
	if errno == 0 {
		switch action {
		case actionMRReg:
			if mrMirror != nil {
				mrHandle := fd.extractMRHandle(buf, int(numAttrs), objectID, rewrites, writeCmdVal)
				if mrHandle != 0 {
					fd.mu.Lock()
					if fd.pinnedMRs == nil {
						fd.pinnedMRs = make(map[uint32]*mirroredPages)
					}
					fd.pinnedMRs[mrHandle] = mrMirror
					fd.mu.Unlock()
					dmaCleanup.Release()
					log.Debugf("rdmaproxy: pinned MR handle=%d (%d ranges)", mrHandle, len(mrMirror.prs))
					if mrMirror.mrSummary != "" {
						log.Infof("rdmaproxy: MR_REG handle=%d %s", mrHandle, mrMirror.mrSummary)
					}
				}
			}

		case actionMRRegDMABuf:
			// Track DMABUF MR handle for DEREG cleanup. No mirrored
			// pages to release — store a nil sentinel so DEREG doesn't
			// warn about missing handles.
			mrHandle := fd.extractDMABufMRHandle(buf, int(numAttrs))
			if mrHandle != 0 {
				fd.mu.Lock()
				if fd.pinnedMRs == nil {
					fd.pinnedMRs = make(map[uint32]*mirroredPages)
				}
				fd.pinnedMRs[mrHandle] = &mirroredPages{}
				fd.mu.Unlock()
				log.Infof("rdmaproxy: DMABUF MR_REG handle=%d hostFD=%d (%s)", mrHandle, fd.hostFD, taskLogFields(t))
			}

		case actionMRDereg:
			mrHandle := fd.extractDeregMRHandle(buf, int(numAttrs), objectID, rewrites, writeCmdVal)
			if mrHandle != 0 {
				fd.mu.Lock()
				if mp, ok := fd.pinnedMRs[mrHandle]; ok {
					delete(fd.pinnedMRs, mrHandle)
					fd.mu.Unlock()
					mp.release(t)
					log.Debugf("rdmaproxy: unpinned MR handle=%d", mrHandle)
				} else {
					fd.mu.Unlock()
				}
			}

		case actionCQCreate:
			if cqqpMirror != nil {
				handle := fd.extractCQQPHandle(buf, int(numAttrs), objectID, rewrites)
				if handle != 0 {
					fd.mu.Lock()
					if fd.pinnedCQs == nil {
						fd.pinnedCQs = make(map[uint32]*pinnedDMABufs)
					}
					fd.pinnedCQs[handle] = cqqpMirror
					fd.mu.Unlock()
					dmaCleanup.Release()
					log.Debugf("rdmaproxy: pinned CQ handle=%d", handle)
				}
			}

		case actionQPCreate:
			if cqqpMirror != nil {
				handle := fd.extractCQQPHandle(buf, int(numAttrs), objectID, rewrites)
				if handle != 0 {
					fd.mu.Lock()
					if fd.pinnedQPs == nil {
						fd.pinnedQPs = make(map[uint32]*pinnedDMABufs)
					}
					fd.pinnedQPs[handle] = cqqpMirror
					fd.mu.Unlock()
					dmaCleanup.Release()
					log.Debugf("rdmaproxy: pinned QP handle=%d", handle)
				}
			}

		case actionCQDestroy:
			handle := fd.extractCQQPDestroyHandle(buf, int(numAttrs), objectID, rewrites)
			if handle != 0 {
				fd.mu.Lock()
				if p, ok := fd.pinnedCQs[handle]; ok {
					delete(fd.pinnedCQs, handle)
					fd.mu.Unlock()
					p.release(t)
					log.Debugf("rdmaproxy: unpinned CQ handle=%d", handle)
				} else {
					fd.mu.Unlock()
				}
			}

		case actionQPDestroy:
			handle := fd.extractCQQPDestroyHandle(buf, int(numAttrs), objectID, rewrites)
			if handle != 0 {
				fd.mu.Lock()
				if p, ok := fd.pinnedQPs[handle]; ok {
					delete(fd.pinnedQPs, handle)
					fd.mu.Unlock()
					p.release(t)
					log.Debugf("rdmaproxy: unpinned QP handle=%d", handle)
				} else {
					fd.mu.Unlock()
				}
			}
		}

		// Proxy the async event FD returned by ASYNC_EVENT_ALLOC.
		// The kernel created this FD in the sentry's host process;
		// we must wrap it so the sandbox can read() async events.
		if objectID == uverbsObjectAsyncEvent && methodID == uverbsMethodAsyncEventAlloc {
			if sentryFD, err := fd.proxyAsyncEventFD(t, buf, int(numAttrs)); err != nil {
				log.Warningf("rdmaproxy: async event FD proxy: %v", err)
			} else if sentryFD >= 0 {
				log.Infof("rdmaproxy: installed async event FD → sandbox fd %d", sentryFD)
			}
		}
	}

	// Copy output data back and restore original pointers.
	for _, rw := range rewrites {
		t.CopyOutBytes(hostarch.Addr(rw.origData), rw.sentry)
		binary.LittleEndian.PutUint64(buf[rw.attrOff+8:rw.attrOff+16], rw.origData)
	}
	t.CopyOutBytes(argPtr, buf)

	if errno != 0 {
		return n, errno
	}
	return n, nil
}

// classifyIoctl determines what DMA-relevant action this ioctl represents.
func (fd *uverbsFD) classifyIoctl(buf []byte, numAttrs int, objectID, methodID uint16) (action ioctlAction, writeCmdVal uint64) {
	// Modern path: direct object methods.
	switch objectID {
	case uverbsObjectMR:
		if methodID == uverbsMethodRegMR {
			return actionMRReg, 0
		}
		if methodID == uverbsMethodRegDMABufMR {
			return actionMRRegDMABuf, 0
		}
		if methodID == uverbsMethodMRDestroy {
			return actionMRDereg, 0
		}
	case uverbsObjectCQ:
		// Method IDs vary across kernel versions, so detect CREATE vs
		// DESTROY by the presence of the mlx5 driver input attr.
		if hasAttrID(buf, numAttrs, mlx5DriverAttrIn) {
			return actionCQCreate, 0
		}
		return actionCQDestroy, 0
	case uverbsObjectQP:
		if hasAttrID(buf, numAttrs, mlx5DriverAttrIn) {
			return actionQPCreate, 0
		}
		return actionQPDestroy, 0
	}

	// Legacy path: INVOKE_WRITE on DEVICE object.
	if objectID == uverbsObjectDevice && methodID == uverbsMethodInvokeWrite {
		writeCmdVal = findInlineAttr(buf, numAttrs, uverbsAttrWriteCmd)
		switch writeCmdVal {
		case ibUserVerbsCmdRegMR:
			return actionMRReg, writeCmdVal
		case ibUserVerbsCmdDeregMR:
			return actionMRDereg, writeCmdVal
		case ibUserVerbsCmdCreateCQ:
			return actionCQCreate, writeCmdVal
		case ibUserVerbsCmdCreateQP:
			return actionQPCreate, writeCmdVal
		case ibUserVerbsCmdDestroyCQ:
			return actionCQDestroy, writeCmdVal
		case ibUserVerbsCmdDestroyQP:
			return actionQPDestroy, writeCmdVal
		default:
			// Preserve writeCmdVal for ioctlNeedsHostNetns (e.g. MODIFY_QP with
			// IB_USER_VERBS_CMD_FLAG_EXTENDED). Previously we returned (actionNone, 0).
			return actionNone, writeCmdVal
		}
	}
	return actionNone, 0
}

// hasAttrID returns true if any attr in the ioctl buffer has the given ID.
func hasAttrID(buf []byte, numAttrs int, targetID uint16) bool {
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == targetID {
			return true
		}
	}
	return false
}

// findInlineAttr finds an attr by ID where CopyIn failed (inline data) and
// returns its data field value. Returns 0 if not found.
func findInlineAttr(buf []byte, numAttrs int, targetID uint16) uint64 {
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		attrLen := binary.LittleEndian.Uint16(buf[off+2 : off+4])
		if attrID == targetID {
			// Inline attrs have len=0 or small len with non-pointer data.
			// The data field is the value itself.
			_ = attrLen
			return binary.LittleEndian.Uint64(buf[off+8 : off+16])
		}
	}
	return 0
}

// findRewrite finds the rewrite entry for a given attr ID.
func findRewrite(buf []byte, numAttrs int, rewrites []attrRewrite, targetID uint16) *attrRewrite {
	for i := range rewrites {
		off := rewrites[i].attrOff
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == targetID {
			return &rewrites[i]
		}
	}
	return nil
}

// prepareMRReg detects the MR address in the ioctl and mirrors the sandbox
// pages into the sentry's address space. The address in the ioctl buffer is
// rewritten to the sentry-side mapping.
func (fd *uverbsFD) prepareMRReg(t *kernel.Task, buf []byte, numAttrs int, objectID uint16, rewrites []attrRewrite, writeCmdVal uint64) (*mirroredPages, error) {
	if objectID == uverbsObjectDevice {
		return fd.prepareMRRegInvokeWrite(t, buf, numAttrs, rewrites)
	}
	return fd.prepareMRRegModern(t, buf, numAttrs, rewrites)
}

// prepareMRRegInvokeWrite handles MR REG via the INVOKE_WRITE legacy path.
// The CORE_IN attr contains an ib_uverbs_reg_mr struct with start/length.
func (fd *uverbsFD) prepareMRRegInvokeWrite(t *kernel.Task, buf []byte, numAttrs int, rewrites []attrRewrite) (*mirroredPages, error) {
	rw := findRewrite(buf, numAttrs, rewrites, uverbsAttrCoreIn)
	if rw == nil {
		log.Warningf("rdmaproxy: MR REG INVOKE_WRITE but no CORE_IN attr found")
		return nil, nil
	}
	if len(rw.sentry) < regMROffHcaVA+8 {
		log.Warningf("rdmaproxy: MR REG CORE_IN too short: %d bytes", len(rw.sentry))
		return nil, nil
	}

	sandboxVA := binary.LittleEndian.Uint64(rw.sentry[regMROffStart : regMROffStart+8])
	length := binary.LittleEndian.Uint64(rw.sentry[regMROffLength : regMROffLength+8])

	log.Debugf("rdmaproxy: MR REG (INVOKE_WRITE) sandbox_va=%#x length=%d", sandboxVA, length)

	if length == 0 {
		return nil, nil
	}

	mp, sentryVA, err := mirrorSandboxPages(t, sandboxVA, length)
	if err != nil {
		return nil, fmt.Errorf("mirrorSandboxPages: %w", err)
	}

	oldHCAVA := binary.LittleEndian.Uint64(rw.sentry[regMROffHcaVA : regMROffHcaVA+8])

	// Rewrite start to sentry address. With identity-mapped GPU VAs,
	// sentryVA == sandboxVA for GPU memory, so hca_va stays unchanged.
	binary.LittleEndian.PutUint64(rw.sentry[regMROffStart:regMROffStart+8], uint64(sentryVA))
	log.Debugf("rdmaproxy: MR REG rewrote start %#x → sentry %#x (hca_va=%#x)",
		sandboxVA, sentryVA, oldHCAVA)
	if mp != nil {
		mp.mrSummary = formatMRSummary(t, sandboxVA, length, sentryVA, oldHCAVA, oldHCAVA, 0, 0)
	}

	return mp, nil
}

// prepareMRRegModern handles MR REG via the modern UVERBS_METHOD_REG_MR path.
// The ADDR and LENGTH attrs carry the values.
func (fd *uverbsFD) prepareMRRegModern(t *kernel.Task, buf []byte, numAttrs int, rewrites []attrRewrite) (*mirroredPages, error) {
	addrRW := findRewrite(buf, numAttrs, rewrites, uverbsAttrRegMRAddr)
	lengthRW := findRewrite(buf, numAttrs, rewrites, uverbsAttrRegMRLength)
	if addrRW == nil || lengthRW == nil {
		// ADDR and LENGTH might be inline for small values.
		// Try reading them as inline values from the raw buffer.
		addrInline := findInlineAttr(buf, numAttrs, uverbsAttrRegMRAddr)
		lengthInline := findInlineAttr(buf, numAttrs, uverbsAttrRegMRLength)
		log.Warningf("rdmaproxy: MR REG (modern) ADDR/LENGTH not rewritten (addrRW=%v lenRW=%v inline addr=%#x len=%d) — forwarding without mirroring",
			addrRW != nil, lengthRW != nil, addrInline, lengthInline)
		return nil, nil
	}
	if len(addrRW.sentry) < 8 || len(lengthRW.sentry) < 8 {
		return nil, nil
	}

	sandboxVA := binary.LittleEndian.Uint64(addrRW.sentry[0:8])
	length := binary.LittleEndian.Uint64(lengthRW.sentry[0:8])

	log.Debugf("rdmaproxy: MR REG (modern) sandbox_va=%#x length=%d", sandboxVA, length)

	if length == 0 {
		return nil, nil
	}

	mp, sentryVA, err := mirrorSandboxPages(t, sandboxVA, length)
	if err != nil {
		return nil, fmt.Errorf("mirrorSandboxPages: %w", err)
	}

	// Rewrite ADDR to sentry address. With identity-mapped GPU VAs,
	// sentryVA == sandboxVA for GPU memory, so IOVA stays unchanged.
	binary.LittleEndian.PutUint64(addrRW.sentry[0:8], uint64(sentryVA))
	log.Debugf("rdmaproxy: MR REG (modern) rewrote addr %#x → sentry %#x", sandboxVA, sentryVA)
	if mp != nil {
		mp.mrSummary = formatMRSummary(t, sandboxVA, length, sentryVA, 0, 0, sandboxVA, sandboxVA)
	}

	return mp, nil
}

// mirrorSandboxPages pins the sandbox pages backing [addr, addr+length) and
// maps them into the sentry's address space. Returns the mirrored pages and
// the sentry VA corresponding to the original sandbox address.
//
// Modeled on nvproxy's rmAllocOSDescriptor.
func mirrorSandboxPages(t *kernel.Task, addr, length uint64) (*mirroredPages, uintptr, error) {
	alignedStart := hostarch.Addr(addr).RoundDown()
	alignedEnd, ok := hostarch.Addr(addr + length).RoundUp()
	if !ok {
		return nil, 0, linuxerr.EINVAL
	}
	alignedLen := uint64(alignedEnd - alignedStart)

	appAR, ok := alignedStart.ToRange(alignedLen)
	if !ok {
		return nil, 0, linuxerr.EINVAL
	}

	// GPU device memory must be identity-mapped at the GPU VA for
	// nvidia-peermem. Check BEFORE Pin because some GPU VAs have
	// CPU-accessible UVM pages (Pin succeeds) but still need GPU VA.
	mp, sentryVA, gpuErr := mirrorGPUDeviceMemory(t, addr, alignedStart, alignedLen)
	if gpuErr == nil {
		return mp, sentryVA, nil
	}
	log.Warningf("rdmaproxy: GPU mirror for %#x failed: %v (falling through to Pin)", addr, gpuErr)

	at := hostarch.ReadWrite
	prs, pinErr := t.MemoryManager().Pin(t, appAR, at, false /* ignorePermissions */)
	if pinErr != nil {
		// Pin fails → must be GPU device memory. GPU mirror already
		// failed above, so return that error.
		return nil, 0, fmt.Errorf("GPU mirror failed (%v) and Pin failed (%v) for %#x", gpuErr, pinErr, addr)
	}

	cu := cleanup.Make(func() { mm.Unpin(prs) })
	defer cu.Clean()

	// Try to get a single contiguous internal mapping.
	var m uintptr
	mOwned := false
	if len(prs) == 1 {
		pr := prs[0]
		ims, err := pr.File.MapInternal(memmap.FileRange{pr.Offset, pr.Offset + uint64(pr.Source.Length())}, at)
		if err == nil && ims.NumBlocks() == 1 {
			m = ims.Head().Addr()
		}
	}

	// If not contiguous, build a contiguous sentry mapping via mmap+mremap.
	if m == 0 {
		var errno unix.Errno
		m, _, errno = unix.RawSyscall6(unix.SYS_MMAP, 0, uintptr(alignedLen), unix.PROT_NONE, unix.MAP_PRIVATE|unix.MAP_ANONYMOUS, ^uintptr(0), 0)
		if errno != 0 {
			return nil, 0, fmt.Errorf("mmap anon %d bytes: %w", alignedLen, errno)
		}
		mOwned = true
		cu.Add(func() {
			unix.RawSyscall(unix.SYS_MUNMAP, m, uintptr(alignedLen), 0)
		})
		sentryAddr := m
		for _, pr := range prs {
			ims, err := pr.File.MapInternal(memmap.FileRange{pr.Offset, pr.Offset + uint64(pr.Source.Length())}, at)
			if err != nil {
				return nil, 0, fmt.Errorf("MapInternal: %w", err)
			}
			for !ims.IsEmpty() {
				im := ims.Head()
				if _, _, errno := unix.RawSyscall6(unix.SYS_MREMAP, im.Addr(), 0, uintptr(im.Len()), linux.MREMAP_MAYMOVE|linux.MREMAP_FIXED, sentryAddr, 0); errno != 0 {
					return nil, 0, fmt.Errorf("mremap %#x→%#x len %d: %w", im.Addr(), sentryAddr, im.Len(), errno)
				}
				sentryAddr += uintptr(im.Len())
				ims = ims.Tail()
			}
		}
	}

	// Best-effort pre-fault to avoid mmap_lock contention.
	unix.Syscall(unix.SYS_MADVISE, m, uintptr(alignedLen), unix.MADV_POPULATE_WRITE)

	mp = &mirroredPages{prs: prs}
	if mOwned {
		mp.m = m
		mp.len = uintptr(alignedLen)
	}
	cu.Release()

	sentryVA = m + uintptr(addr-uint64(alignedStart))
	return mp, sentryVA, nil
}

// mirrorGPUDeviceMemory creates a nvidia-backed VMA at the GPU VA. If a VMA
// already exists at this address (from a previous MR_REG), reuse it to avoid
// clobbering the nvidia mmap context with MAP_FIXED. Multiple MRs at the
// same GPU VA share the VMA via refcounting.
func mirrorGPUDeviceMemory(t *kernel.Task, addr uint64, alignedStart hostarch.Addr, alignedLen uint64) (*mirroredPages, uintptr, error) {
	tgid := int32(t.TGIDInRoot())

	// Check cache — only reuse VMAs from the SAME tgid (nvidia RM context
	// is per-client, VMAs can't be shared across tgids).
	if v := acquireGPUVMA(tgid, uintptr(alignedStart), uintptr(alignedLen)); v != nil {
		return &mirroredPages{gpuVMA: v, gpuVMACached: true}, uintptr(alignedStart), nil
	}

	// Collect frontends: same-tgid first from registry, then cross-tgid, then FD scan.
	frontends := lookupAllGPUVA(tgid, addr)
	if len(frontends) == 0 {
		seen := make(map[GPUVAFrontend]bool)
		t.FDTable().ForEach(t, func(fd int32, file *vfs.FileDescription, _ kernel.FDFlags) bool {
			if fe, ok := file.Impl().(GPUVAFrontend); ok && !seen[fe] {
				seen[fe] = true
				frontends = append(frontends, fe)
			}
			return true
		})
	}
	if len(frontends) == 0 {
		return nil, 0, fmt.Errorf("no frontendFD for GPU VA %#x", addr)
	}

	// Serialize RM_MAP_MEMORY → mmap → cache under lock.
	gpuVMACache.mu.Lock()

	// Re-check cache under lock (same tgid only).
	if gpuVMACache.byKey != nil {
		reqEnd := uintptr(alignedStart) + uintptr(alignedLen)
		for key, existing := range gpuVMACache.byKey {
			if key.tgid != tgid {
				continue
			}
			vEnd := existing.va + existing.len
			if uintptr(alignedStart) >= existing.va && reqEnd <= vEnd {
				existing.refs++
				gpuVMACache.mu.Unlock()
				return &mirroredPages{gpuVMA: existing, gpuVMACached: true}, existing.va, nil
			}
		}
	}

	var lastErr error
	for _, fe := range frontends {
		mapFD, devName, _, err := fe.NVProxyPrepareGPUVMA(t, addr, uint64(alignedStart), alignedLen, uint64(alignedStart))
		if err != nil {
			lastErr = err
			continue
		}
		mapped, _, mmapErrno := unix.RawSyscall6(unix.SYS_MMAP,
			uintptr(alignedStart), uintptr(alignedLen),
			unix.PROT_READ|unix.PROT_WRITE,
			unix.MAP_SHARED|unix.MAP_FIXED,
			uintptr(mapFD), 0)
		if mmapErrno != 0 {
			lastErr = fmt.Errorf("mmap at %#x: %v", alignedStart, mmapErrno)
			continue
		}
		if gpuVMACache.byKey == nil {
			gpuVMACache.byKey = make(map[gpuVMAKey]*gpuVMARef)
		}
		key := gpuVMAKey{va: mapped, tgid: tgid}
		v := &gpuVMARef{va: mapped, len: uintptr(alignedLen), tgid: tgid, refs: 1}
		gpuVMACache.byKey[key] = v
		gpuVMACache.mu.Unlock()
		log.Warningf("rdmaproxy: mirrorGPUDeviceMemory %#x: NEW VMA at %#x-%#x dev=%q mapFD=%d tgid=%d (%s)",
			addr, alignedStart, uint64(alignedStart)+alignedLen, devName, mapFD, tgid, taskLogFields(t))
		return &mirroredPages{gpuVMA: v}, mapped, nil
	}
	gpuVMACache.mu.Unlock()
	return nil, 0, fmt.Errorf("all %d frontends failed for GPU VA %#x: %w", len(frontends), addr, lastErr)
}

// extractMRHandle reads the MR handle from the ioctl response after
// a successful MR REG.
func (fd *uverbsFD) extractMRHandle(buf []byte, numAttrs int, objectID uint16, rewrites []attrRewrite, writeCmdVal uint64) uint32 {
	if objectID == uverbsObjectDevice {
		// INVOKE_WRITE: response in CORE_OUT attr.
		rw := findRewrite(buf, numAttrs, rewrites, uverbsAttrCoreOut)
		if rw == nil || len(rw.sentry) < regMRRespOffHandle+4 {
			log.Warningf("rdmaproxy: MR REG success but no CORE_OUT to read handle")
			return 0
		}
		return binary.LittleEndian.Uint32(rw.sentry[regMRRespOffHandle : regMRRespOffHandle+4])
	}
	// Modern path: HANDLE attr (id=0) is an output IDR — the handle is
	// returned in the data field of the attr in the response buffer.
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == uverbsAttrRegMRHandle {
			return uint32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		}
	}
	return 0
}

// rewriteDMABufFD translates the DMA-BUF fd in a DMABUF MR_REG ioctl
// from the sandbox's fd number to the host fd. The fd is attr 5
// (UVERBS_ATTR_REG_DMABUF_MR_FD), stored inline (attrLen == 0).
func (fd *uverbsFD) rewriteDMABufFD(t *kernel.Task, buf []byte, numAttrs int) {
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		attrLen := binary.LittleEndian.Uint16(buf[off+2 : off+4])
		if attrID != uverbsAttrRegDMABufMRFD || attrLen != 0 {
			continue
		}
		sandboxFD := int32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		if sandboxFD < 0 {
			// fd=-1 is the NCCL probe — let it through as-is.
			log.Debugf("rdmaproxy: DMABUF MR_REG fd=%d (probe), forwarding as-is", sandboxFD)
			return
		}
		file, _ := t.FDTable().Get(sandboxFD)
		if file == nil {
			log.Warningf("rdmaproxy: DMABUF MR_REG sandbox fd=%d not found in fd table", sandboxFD)
			return
		}
		defer file.DecRef(t)
		// Try all known fd types that wrap a host fd.
		if hostFDer, ok := file.Impl().(interface{ NVProxyHostFD() int32 }); ok {
			hostFD := hostFDer.NVProxyHostFD()
			binary.LittleEndian.PutUint64(buf[off+8:off+16], uint64(hostFD))
			log.Debugf("rdmaproxy: DMABUF MR_REG fd rewrite sandbox=%d → host=%d (nvproxy)", sandboxFD, hostFD)
			return
		}
		log.Warningf("rdmaproxy: DMABUF MR_REG sandbox fd=%d is type %T — cannot translate to host fd (nvproxy DMA-BUF export not yet supported)", sandboxFD, file.Impl())
		return
	}
}

// extractDMABufMRHandle reads the MR handle from a DMABUF MR_REG response.
// The handle is attr 0 (UVERBS_ATTR_REG_DMABUF_MR_HANDLE), inline.
func (fd *uverbsFD) extractDMABufMRHandle(buf []byte, numAttrs int) uint32 {
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == uverbsAttrRegDMABufMRHandle {
			return uint32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		}
	}
	return 0
}

// extractDeregMRHandle reads the MR handle from a DEREG_MR ioctl.
func (fd *uverbsFD) extractDeregMRHandle(buf []byte, numAttrs int, objectID uint16, rewrites []attrRewrite, writeCmdVal uint64) uint32 {
	if objectID == uverbsObjectDevice {
		// INVOKE_WRITE: mr_handle in CORE_IN attr.
		rw := findRewrite(buf, numAttrs, rewrites, uverbsAttrCoreIn)
		if rw == nil || len(rw.sentry) < deregMROffHandle+4 {
			return 0
		}
		return binary.LittleEndian.Uint32(rw.sentry[deregMROffHandle : deregMROffHandle+4])
	}
	// Modern path: DESTROY_MR_HANDLE attr (id=0).
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == uverbsAttrDestroyMRHandle {
			return uint32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		}
	}
	return 0
}

// prepareCQQPCreate mirrors the DMA buffers (buf_addr + db_addr) embedded in
// the mlx5 driver attribute of a CQ or QP CREATE ioctl. Uses FindVMARange to
// determine buffer sizes from the mmap region boundaries.
func (fd *uverbsFD) prepareCQQPCreate(t *kernel.Task, buf []byte, numAttrs int, rewrites []attrRewrite, action ioctlAction) (*pinnedDMABufs, error) {
	drv := findRewrite(buf, numAttrs, rewrites, mlx5DriverAttrIn)
	if drv == nil {
		log.Debugf("rdmaproxy: CQ/QP CREATE but no driver attr 0x%x found", mlx5DriverAttrIn)
		return nil, nil
	}
	if len(drv.sentry) < driverAttrMinLen {
		log.Debugf("rdmaproxy: CQ/QP CREATE driver attr too short: %d bytes", len(drv.sentry))
		return nil, nil
	}

	bufAddr := binary.LittleEndian.Uint64(drv.sentry[driverAttrBufAddr : driverAttrBufAddr+8])
	dbAddr := binary.LittleEndian.Uint64(drv.sentry[driverAttrDBAddr : driverAttrDBAddr+8])
	kind := "CQ"
	if action == actionQPCreate {
		kind = "QP"
	}
	log.Debugf("rdmaproxy: %s CREATE buf_addr=%#x db_addr=%#x", kind, bufAddr, dbAddr)

	var bufs pinnedDMABufs
	var cu cleanup.Cleanup
	defer cu.Clean()

	if bufAddr != 0 {
		vmaRange, err := t.MemoryManager().FindVMARange(hostarch.Addr(bufAddr))
		if err != nil {
			return nil, fmt.Errorf("FindVMARange(buf %#x): %w", bufAddr, err)
		}
		length := uint64(vmaRange.End) - bufAddr
		mp, sentryVA, err := mirrorSandboxPages(t, bufAddr, length)
		if err != nil {
			return nil, fmt.Errorf("mirrorSandboxPages buf: %w", err)
		}
		bufs.buf = mp
		cu.Add(func() { mp.release(t) })
		binary.LittleEndian.PutUint64(drv.sentry[driverAttrBufAddr:driverAttrBufAddr+8], uint64(sentryVA))
		log.Debugf("rdmaproxy: %s CREATE buf %#x → sentry %#x (len=%d)", kind, bufAddr, sentryVA, length)
	}

	if dbAddr != 0 {
		vmaRange, err := t.MemoryManager().FindVMARange(hostarch.Addr(dbAddr))
		if err != nil {
			return nil, fmt.Errorf("FindVMARange(db %#x): %w", dbAddr, err)
		}
		length := uint64(vmaRange.End) - dbAddr
		mp, sentryVA, err := mirrorSandboxPages(t, dbAddr, length)
		if err != nil {
			return nil, fmt.Errorf("mirrorSandboxPages db: %w", err)
		}
		bufs.db = mp
		cu.Add(func() { mp.release(t) })
		binary.LittleEndian.PutUint64(drv.sentry[driverAttrDBAddr:driverAttrDBAddr+8], uint64(sentryVA))
		log.Debugf("rdmaproxy: %s CREATE db %#x → sentry %#x (len=%d)", kind, dbAddr, sentryVA, length)
	}

	cu.Release()
	return &bufs, nil
}

// extractCQQPHandle reads the CQ or QP handle from the ioctl response after
// a successful CREATE.
func (fd *uverbsFD) extractCQQPHandle(buf []byte, numAttrs int, objectID uint16, rewrites []attrRewrite) uint32 {
	if objectID == uverbsObjectDevice {
		// INVOKE_WRITE: handle is first __u32 of CORE_OUT.
		rw := findRewrite(buf, numAttrs, rewrites, uverbsAttrCoreOut)
		if rw == nil || len(rw.sentry) < 4 {
			return 0
		}
		return binary.LittleEndian.Uint32(rw.sentry[0:4])
	}
	// Modern path: handle in attr id=0 data field (IDR output, written by kernel).
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == 0 {
			return uint32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		}
	}
	return 0
}

// extractCQQPDestroyHandle reads the CQ or QP handle from a DESTROY ioctl.
// INVOKE_WRITE path: handle is first __u32 of CORE_IN.
// Modern path: handle is in attr id=0 data field.
func (fd *uverbsFD) extractCQQPDestroyHandle(buf []byte, numAttrs int, objectID uint16, rewrites []attrRewrite) uint32 {
	if objectID == uverbsObjectDevice {
		rw := findRewrite(buf, numAttrs, rewrites, uverbsAttrCoreIn)
		if rw == nil || len(rw.sentry) < 4 {
			return 0
		}
		return binary.LittleEndian.Uint32(rw.sentry[0:4])
	}
	// Modern path: attr id=0 contains the handle.
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID == 0 {
			return uint32(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		}
	}
	return 0
}

// proxyAsyncEventFD extracts the host FD from a successful
// ASYNC_EVENT_ALLOC response, wraps it in a sentry FileDescription,
// installs it in the task's FD table, and rewrites the ioctl buffer
// so the sandbox receives the proxy FD number.
func (fd *uverbsFD) proxyAsyncEventFD(t *kernel.Task, buf []byte, numAttrs int) (int32, error) {
	for i := 0; i < numAttrs; i++ {
		off := ibUverbsIoctlHdrSize + i*ibUverbsAttrSize
		attrID := binary.LittleEndian.Uint16(buf[off : off+2])
		if attrID != uverbsAttrAsyncEventAllocFD {
			continue
		}
		hostFD := int(binary.LittleEndian.Uint64(buf[off+8 : off+16]))
		if hostFD < 0 {
			return -1, fmt.Errorf("kernel returned invalid async event fd %d", hostFD)
		}
		sentryFD, err := newAsyncEventFD(t, hostFD)
		if err != nil {
			unix.Close(hostFD)
			return -1, fmt.Errorf("newAsyncEventFD: %w", err)
		}
		// No global map needed — the FD rewrite loop now resolves
		// sandbox FDs through the task's FD table at ioctl time.
		binary.LittleEndian.PutUint64(buf[off+8:off+16], uint64(sentryFD))
		return sentryFD, nil
	}
	return -1, fmt.Errorf("ASYNC_EVENT_ALLOC response missing FD attr")
}

// Write implements vfs.FileDescriptionImpl.Write.
// This handles the legacy uverbs write() command interface where rdma-core
// sends commands like ALLOC_PD, REG_MR, DEREG_MR via write() on the fd.
func (fd *uverbsFD) Write(ctx context.Context, src usermem.IOSequence, opts vfs.WriteOptions) (int64, error) {
	globalWriteCount.Add(1)

	t := kernel.TaskFromContext(ctx)
	if t == nil {
		return 0, linuxerr.EINVAL
	}

	size := src.NumBytes()
	if size < 8 {
		return 0, linuxerr.EINVAL
	}

	data := make([]byte, size)
	if _, err := src.CopyIn(ctx, data); err != nil {
		return 0, err
	}

	rawCmd := binary.LittleEndian.Uint32(data[0:4])
	cmdBase := rawCmd & 0x7FFFFFFF
	isExtended := rawCmd&0x80000000 != 0
	inWords := binary.LittleEndian.Uint16(data[4:6])
	outWords := binary.LittleEndian.Uint16(data[6:8])

	log.Debugf("rdmaproxy: Write cmd=%d extended=%v in_words=%d out_words=%d len=%d",
		cmdBase, isExtended, inWords, outWords, size)
	startPerfReporter()
	if !isExtended {
		countAction(actionFromLegacyWriteCmd(cmdBase))
	}

	// Response pointer is always at byte offset 8 (first field of the
	// command-specific struct for non-extended, or the ex_hdr for extended).
	// Rewrite it to a sentry-side buffer so the host kernel's copy_to_user
	// writes into our address space rather than the sandbox.
	var origResp uint64
	var respBuf []byte
	if outWords > 0 && size >= 16 {
		origResp = binary.LittleEndian.Uint64(data[8:16])
		respLen := int(outWords) * 4
		respBuf = make([]byte, respLen)
		binary.LittleEndian.PutUint64(data[8:16],
			uint64(uintptr(unsafe.Pointer(&respBuf[0]))))
		log.Debugf("rdmaproxy: Write cmd=%d resp rewrite %#x → sentry (%d bytes)",
			cmdBase, origResp, respLen)
	}

	// Mirror DMA pages for commands that need sentry-side pinning.
	var mrMirror *mirroredPages
	var cqqpMirror *pinnedDMABufs
	var cu cleanup.Cleanup
	defer cu.Clean()

	if !isExtended {
		switch cmdBase {
		case ibUserVerbsCmdRegMR:
			// Non-extended ib_uverbs_reg_mr layout after 8-byte cmd_hdr:
			//   +0: response (8)  +8: start (8)  +16: length (8)  +24: hca_va (8)
			const startOff, lengthOff, hcaVAOff = 16, 24, 32
			if size >= hcaVAOff+8 {
				sva := binary.LittleEndian.Uint64(data[startOff : startOff+8])
				length := binary.LittleEndian.Uint64(data[lengthOff : lengthOff+8])
				log.Debugf("rdmaproxy: Write REG_MR va=%#x len=%d", sva, length)

				if length > 0 {
					mp, sentryVA, err := mirrorSandboxPages(t, sva, length)
					if err != nil {
						log.Warningf("rdmaproxy: Write REG_MR mirrorSandboxPages: %v", err)
						return 0, linuxerr.ENOMEM
					}
					mrMirror = mp
					cu = cleanup.Make(func() { mp.release(t) })
					binary.LittleEndian.PutUint64(data[startOff:startOff+8], uint64(sentryVA))
					log.Debugf("rdmaproxy: Write REG_MR rewrite start %#x → sentry %#x (hca_va=%#x)",
						sva, sentryVA,
						binary.LittleEndian.Uint64(data[hcaVAOff:hcaVAOff+8]))
					if mp != nil {
						hcaVA := binary.LittleEndian.Uint64(data[hcaVAOff : hcaVAOff+8])
						mp.mrSummary = formatMRSummary(t, sva, length, sentryVA, hcaVA, hcaVA, 0, 0)
					}
				}
			}

		case ibUserVerbsCmdCreateCQ, ibUserVerbsCmdCreateQP:
			cqqpMirror = fd.prepareLegacyCQQPCreate(t, data, cmdBase)
			if cqqpMirror != nil {
				cu = cleanup.Make(func() { cqqpMirror.release(t) })
			}
		}
	}

	// Forward write to host fd.
	var n uintptr
	var errno unix.Errno
	if !isExtended && cmdBase == ibUserVerbsCmdModifyQP {
		n, errno = writeInHostNetns(fd.hostFD, data)
	} else {
		n, _, errno = unix.RawSyscall(unix.SYS_WRITE,
			uintptr(fd.hostFD),
			uintptr(unsafe.Pointer(&data[0])),
			uintptr(size))
	}
	if errno != 0 {
		log.Warningf("rdmaproxy: Write to host: n=%d errno=%d (%v)", n, errno, errno)
		return 0, errno
	}
	log.Debugf("rdmaproxy: Write to host returned %d OK (cmd=%d)", n, cmdBase)

	// Copy response back to sandbox.
	if respBuf != nil && origResp != 0 {
		if _, err := t.CopyOutBytes(hostarch.Addr(origResp), respBuf); err != nil {
			log.Warningf("rdmaproxy: Write response CopyOut to %#x: %v", origResp, err)
		}
		binary.LittleEndian.PutUint64(data[8:16], origResp)
	}

	// Post-write tracking for successful operations.
	if errno == 0 && !isExtended {
		switch cmdBase {
		case ibUserVerbsCmdRegMR:
			if mrMirror != nil && respBuf != nil && len(respBuf) >= 4 {
				mrHandle := binary.LittleEndian.Uint32(respBuf[0:4])
				fd.mu.Lock()
				if fd.pinnedMRs == nil {
					fd.pinnedMRs = make(map[uint32]*mirroredPages)
				}
				fd.pinnedMRs[mrHandle] = mrMirror
				fd.mu.Unlock()
				cu.Release()
				log.Debugf("rdmaproxy: Write REG_MR pinned handle=%d (%d ranges)", mrHandle, len(mrMirror.prs))
				if mrMirror.mrSummary != "" {
					log.Infof("rdmaproxy: Write MR_REG handle=%d %s", mrHandle, mrMirror.mrSummary)
				}
			}

		case ibUserVerbsCmdDeregMR:
			if size >= 12 {
				mrHandle := binary.LittleEndian.Uint32(data[8:12])
				fd.mu.Lock()
				if mp, ok := fd.pinnedMRs[mrHandle]; ok {
					delete(fd.pinnedMRs, mrHandle)
					fd.mu.Unlock()
					mp.release(t)
					log.Debugf("rdmaproxy: Write DEREG_MR unpinned handle=%d", mrHandle)
				} else {
					fd.mu.Unlock()
				}
			}

		case ibUserVerbsCmdCreateCQ:
			if cqqpMirror != nil && respBuf != nil && len(respBuf) >= 4 {
				handle := binary.LittleEndian.Uint32(respBuf[0:4])
				fd.mu.Lock()
				if fd.pinnedCQs == nil {
					fd.pinnedCQs = make(map[uint32]*pinnedDMABufs)
				}
				fd.pinnedCQs[handle] = cqqpMirror
				fd.mu.Unlock()
				cu.Release()
				log.Debugf("rdmaproxy: Write CREATE_CQ pinned handle=%d", handle)
			}

		case ibUserVerbsCmdCreateQP:
			if cqqpMirror != nil && respBuf != nil && len(respBuf) >= 4 {
				handle := binary.LittleEndian.Uint32(respBuf[0:4])
				fd.mu.Lock()
				if fd.pinnedQPs == nil {
					fd.pinnedQPs = make(map[uint32]*pinnedDMABufs)
				}
				fd.pinnedQPs[handle] = cqqpMirror
				fd.mu.Unlock()
				cu.Release()
				log.Debugf("rdmaproxy: Write CREATE_QP pinned handle=%d", handle)
			}

		case ibUserVerbsCmdDestroyCQ:
			if size >= 12 {
				handle := binary.LittleEndian.Uint32(data[8:12])
				fd.mu.Lock()
				if p, ok := fd.pinnedCQs[handle]; ok {
					delete(fd.pinnedCQs, handle)
					fd.mu.Unlock()
					p.release(t)
					log.Debugf("rdmaproxy: Write DESTROY_CQ unpinned handle=%d", handle)
				} else {
					fd.mu.Unlock()
				}
			}

		case ibUserVerbsCmdDestroyQP:
			if size >= 12 {
				handle := binary.LittleEndian.Uint32(data[8:12])
				fd.mu.Lock()
				if p, ok := fd.pinnedQPs[handle]; ok {
					delete(fd.pinnedQPs, handle)
					fd.mu.Unlock()
					p.release(t)
					log.Debugf("rdmaproxy: Write DESTROY_QP unpinned handle=%d", handle)
				} else {
					fd.mu.Unlock()
				}
			}
		}
	}

	return int64(n), nil
}

// prepareLegacyCQQPCreate handles CQ/QP CREATE via the legacy write() path.
// The driver data (buf_addr + db_addr) is appended after the core struct in
// the write buffer. Layout: [cmd_hdr (8)] [core_struct] [driver_data...].
func (fd *uverbsFD) prepareLegacyCQQPCreate(t *kernel.Task, data []byte, cmdBase uint32) *pinnedDMABufs {
	// Core struct sizes (after 8-byte cmd_hdr and 8-byte response field):
	//   CREATE_CQ: response(8) + user_handle(8) + cqe(4) + comp_vector(4) + comp_channel(4) + reserved(4) = 32
	//   CREATE_QP: response(8) + user_handle(8) + pd(4) + scq(4) + rcq(4) + srq(4) +
	//              max_send_wr(4) + max_recv_wr(4) + max_send_sge(4) + max_recv_sge(4) +
	//              max_inline(4) + sq_sig_all(1) + qp_type(1) + is_srq(1) + reserved(1) = 56
	var coreSize int
	var kind string
	switch cmdBase {
	case ibUserVerbsCmdCreateCQ:
		coreSize = 32
		kind = "CQ"
	case ibUserVerbsCmdCreateQP:
		coreSize = 56
		kind = "QP"
	default:
		return nil
	}

	drvOff := 8 + coreSize // cmd_hdr + core struct
	if int64(len(data)) < int64(drvOff)+int64(driverAttrMinLen) {
		log.Debugf("rdmaproxy: Write CREATE_%s no driver data (data len=%d, need=%d)", kind, len(data), drvOff+driverAttrMinLen)
		return nil
	}

	bufAddr := binary.LittleEndian.Uint64(data[drvOff : drvOff+8])
	dbAddr := binary.LittleEndian.Uint64(data[drvOff+8 : drvOff+16])
	log.Debugf("rdmaproxy: Write CREATE_%s buf_addr=%#x db_addr=%#x", kind, bufAddr, dbAddr)

	var bufs pinnedDMABufs
	var cu cleanup.Cleanup
	defer cu.Clean()

	if bufAddr != 0 {
		vmaRange, err := t.MemoryManager().FindVMARange(hostarch.Addr(bufAddr))
		if err != nil {
			log.Warningf("rdmaproxy: Write CREATE_%s FindVMARange(buf %#x): %v", kind, bufAddr, err)
			return nil
		}
		length := uint64(vmaRange.End) - bufAddr
		mp, sentryVA, err := mirrorSandboxPages(t, bufAddr, length)
		if err != nil {
			log.Warningf("rdmaproxy: Write CREATE_%s mirrorSandboxPages buf: %v", kind, err)
			return nil
		}
		bufs.buf = mp
		cu.Add(func() { mp.release(t) })
		binary.LittleEndian.PutUint64(data[drvOff:drvOff+8], uint64(sentryVA))
		log.Debugf("rdmaproxy: Write CREATE_%s buf %#x → sentry %#x (len=%d)", kind, bufAddr, sentryVA, length)
	}

	if dbAddr != 0 {
		vmaRange, err := t.MemoryManager().FindVMARange(hostarch.Addr(dbAddr))
		if err != nil {
			log.Warningf("rdmaproxy: Write CREATE_%s FindVMARange(db %#x): %v", kind, dbAddr, err)
			return nil
		}
		length := uint64(vmaRange.End) - dbAddr
		mp, sentryVA, err := mirrorSandboxPages(t, dbAddr, length)
		if err != nil {
			log.Warningf("rdmaproxy: Write CREATE_%s mirrorSandboxPages db: %v", kind, err)
			return nil
		}
		bufs.db = mp
		cu.Add(func() { mp.release(t) })
		binary.LittleEndian.PutUint64(data[drvOff+8:drvOff+16], uint64(sentryVA))
		log.Debugf("rdmaproxy: Write CREATE_%s db %#x → sentry %#x (len=%d)", kind, dbAddr, sentryVA, length)
	}

	cu.Release()
	return &bufs
}

// Read implements vfs.FileDescriptionImpl.Read.
// Forwards reads to the host fd (used for async event notifications).
func (fd *uverbsFD) Read(ctx context.Context, dst usermem.IOSequence, opts vfs.ReadOptions) (int64, error) {
	globalReadCount.Add(1)

	buf := make([]byte, dst.NumBytes())
	n, _, errno := unix.RawSyscall(unix.SYS_READ,
		uintptr(fd.hostFD),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)))
	if errno != 0 {
		if errno == unix.EAGAIN || errno == unix.EWOULDBLOCK {
			return 0, linuxerr.ErrWouldBlock
		}
		return 0, errno
	}
	if n == 0 {
		return 0, nil
	}
	written, err := dst.CopyOut(ctx, buf[:n])
	return int64(written), err
}

// Read implements vfs.FileDescriptionImpl.Read for asyncEventFD.
// Uses fdnotifier to check readiness before reading, so we never
// issue a blocking read() syscall on the host FD.
func (fd *asyncEventFD) Read(ctx context.Context, dst usermem.IOSequence, opts vfs.ReadOptions) (int64, error) {
	globalReadCount.Add(1)
	if fdnotifier.NonBlockingPoll(fd.hostFD, waiter.ReadableEvents) == 0 {
		return 0, linuxerr.ErrWouldBlock
	}
	buf := make([]byte, dst.NumBytes())
	n, _, errno := unix.RawSyscall(unix.SYS_READ,
		uintptr(fd.hostFD),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)))
	if errno != 0 {
		if errno == unix.EAGAIN || errno == unix.EWOULDBLOCK {
			return 0, linuxerr.ErrWouldBlock
		}
		return 0, errno
	}
	if n == 0 {
		return 0, nil
	}
	written, err := dst.CopyOut(ctx, buf[:n])
	return int64(written), err
}

// ConfigureMMap implements vfs.FileDescriptionImpl.ConfigureMMap.
func (fd *uverbsFD) ConfigureMMap(ctx context.Context, opts *memmap.MMapOpts) error {
	log.Debugf("rdmaproxy: mmap hostFD=%d len=%d offset=0x%x perms=%v private=%v",
		fd.hostFD, opts.Length, opts.Offset, opts.Perms, opts.Private)
	err := vfs.GenericProxyDeviceConfigureMMap(&fd.vfsfd, fd, opts)
	if err != nil {
		log.Warningf("rdmaproxy: mmap hostFD=%d: %v", fd.hostFD, err)
	}
	return err
}

// Translate implements memmap.Mappable.Translate.
func (fd *uverbsFD) Translate(ctx context.Context, required, optional memmap.MappableRange, at hostarch.AccessType) ([]memmap.Translation, error) {
	return []memmap.Translation{
		{
			Source: optional,
			File:   &fd.memmapFile,
			Offset: optional.Start,
			Perms:  hostarch.AnyAccess,
		},
	}, nil
}
