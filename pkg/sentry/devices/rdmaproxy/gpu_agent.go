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
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/hostsyscall"
	"gvisor.dev/gvisor/pkg/log"
)

// gpuAgent is a child process with its own mm_struct that holds nvidia-backed
// VMAs for one GPU device. nvidia-peermem requires VMAs at the exact GPU VA
// for nvidia_p2p_get_pages to resolve GPU physical pages. Multiple GPUs can
// allocate at the same GPU VA, so each GPU needs its own address space.
//
// The agent communicates with the sentry via a shared memory page using
// futex-based signaling (no pipes needed — avoids SYS_PIPE2 seccomp issues).
type gpuAgent struct {
	devName string
	pid     int32

	// sharedPage is a MAP_SHARED anonymous page for passing commands,
	// ioctl buffers, and results between sentry and agent. Both processes
	// see the same physical page.
	sharedPage uintptr

	// mu serializes commands to this agent.
	mu sync.Mutex
}

// Shared page layout constants.
const (
	// Command types.
	agentCmdNone         = 0 // idle / done
	agentCmdMmapAndIoctl = 1
	agentCmdMunmap       = 2
	agentCmdExit         = 3

	// Shared page field offsets.
	agentOffCmd      = 0  // u32: command (futex word — 0=idle, nonzero=command)
	agentOffDone     = 4  // u32: completion flag (futex word — 0=busy, 1=done)
	agentOffGPUVA    = 8  // u64: GPU virtual address for mmap
	agentOffLen      = 16 // u64: length for mmap
	agentOffNvidiaFD = 24 // i32: nvidia device FD for mmap
	agentOffUverbsFD = 28 // i32: uverbs host FD for ioctl
	agentOffIoctlCmd = 32 // u32: ioctl command number
	agentOffResultN  = 40 // i64: ioctl return value
	agentOffErrno    = 48 // i32: ioctl errno
	agentOffBufLen   = 52 // u32: ioctl buffer length
	agentOffBuf      = 64 // ioctl buffer data
	agentSharedSize  = 8192
)

// gpuAgents is the global registry of per-GPU agent processes.
var gpuAgents struct {
	mu    sync.Mutex
	byDev map[string]*gpuAgent
}

// getOrSpawnGPUAgent returns the agent for the given GPU device, spawning
// one if it doesn't exist yet.
func getOrSpawnGPUAgent(devName string) (*gpuAgent, error) {
	gpuAgents.mu.Lock()
	defer gpuAgents.mu.Unlock()

	if gpuAgents.byDev == nil {
		gpuAgents.byDev = make(map[string]*gpuAgent)
	}
	if agent := gpuAgents.byDev[devName]; agent != nil {
		return agent, nil
	}

	agent, err := spawnGPUAgent(devName)
	if err != nil {
		return nil, err
	}
	gpuAgents.byDev[devName] = agent
	return agent, nil
}

// spawnGPUAgent creates a new child process for the given GPU device.
func spawnGPUAgent(devName string) (*gpuAgent, error) {
	// Allocate shared memory page.
	sharedPage, _, errno := unix.RawSyscall6(unix.SYS_MMAP,
		0, agentSharedSize,
		unix.PROT_READ|unix.PROT_WRITE,
		unix.MAP_SHARED|unix.MAP_ANONYMOUS,
		^uintptr(0), 0)
	if errno != 0 {
		return nil, fmt.Errorf("mmap shared page: %v", errno)
	}

	// Initialize: cmd=0 (idle), done=0 (not done).
	atomic.StoreUint32((*uint32)(unsafe.Pointer(sharedPage + agentOffCmd)), agentCmdNone)
	atomic.StoreUint32((*uint32)(unsafe.Pointer(sharedPage + agentOffDone)), 0)

	// Clone with CLONE_FILES but without CLONE_VM:
	// - Own mm_struct (no CLONE_VM) → GPU VMAs don't collide
	// - Shared FD table (CLONE_FILES) → agent sees FDs opened after clone
	// CLONE_FILES|SIGCHLD = 0x411, in the systrap seccomp allowlist.
	log.Infof("rdmaproxy: about to clone for GPU agent dev=%q", devName)
	beforeFork()
	pid, errno := hostsyscall.RawSyscall(unix.SYS_CLONE, unix.CLONE_FILES|uintptr(unix.SIGCHLD), 0, 0)
	if errno != 0 {
		afterFork()
		unix.RawSyscall(unix.SYS_MUNMAP, sharedPage, agentSharedSize, 0)
		log.Warningf("rdmaproxy: clone failed for GPU agent dev=%q: errno=%d (%v)", devName, errno, errno)
		return nil, fmt.Errorf("clone: %v", errno)
	}

	if pid == 0 {
		// Child process — raw syscalls only, no Go runtime.
		// Do NOT call afterForkInChild() — with CLONE_FILES it would
		// close Go runtime internal FDs (epoll, pipes) in the shared
		// FD table, breaking the parent's runtime. The child only
		// needs to run raw syscalls, not Go code.
		gpuAgentLoop(sharedPage)
		// unreachable
	}

	// Parent process.
	afterFork()

	agent := &gpuAgent{
		devName:    devName,
		pid:        int32(pid),
		sharedPage: sharedPage,
	}
	log.Infof("rdmaproxy: spawned GPU agent pid=%d for dev=%q", pid, devName)
	return agent, nil
}

// gpuAgentLoop is the child process's main loop. Uses futex on the shared
// page for signaling. Only raw syscalls — Go runtime is not fork-safe.
//
//go:norace
//go:nosplit
func gpuAgentLoop(shared uintptr) {
	cmdPtr := (*uint32)(unsafe.Pointer(shared + agentOffCmd))
	donePtr := (*uint32)(unsafe.Pointer(shared + agentOffDone))

	for {
		// Wait for a command: futex_wait until *cmdPtr != 0.
		// Use raw pointer reads (not sync/atomic) — Go runtime is
		// not safe after fork.
		for {
			cmd := *cmdPtr
			if cmd != agentCmdNone {
				break
			}
			// FUTEX_WAIT: sleep if *cmdPtr is still 0.
			hostsyscall.RawSyscall6(unix.SYS_FUTEX,
				uintptr(unsafe.Pointer(cmdPtr)),
				0, // FUTEX_WAIT
				0, // expected value
				0, 0, 0)
		}

		cmd := *cmdPtr

		switch cmd {
		case agentCmdMmapAndIoctl:
			gpuVA := *(*uint64)(unsafe.Pointer(shared + agentOffGPUVA))
			length := *(*uint64)(unsafe.Pointer(shared + agentOffLen))
			nvidiaFD := *(*int32)(unsafe.Pointer(shared + agentOffNvidiaFD))
			uverbsFD := *(*int32)(unsafe.Pointer(shared + agentOffUverbsFD))
			ioctlCmd := *(*uint32)(unsafe.Pointer(shared + agentOffIoctlCmd))

			// munmap any existing mapping at the target GPU VA.
			hostsyscall.RawSyscall(unix.SYS_MUNMAP, uintptr(gpuVA), uintptr(length), 0)

			// mmap nvidia FD at the exact GPU VA.
			_, mmapErrno := hostsyscall.RawSyscall6(unix.SYS_MMAP,
				uintptr(gpuVA), uintptr(length),
				unix.PROT_READ|unix.PROT_WRITE,
				unix.MAP_SHARED|unix.MAP_FIXED,
				uintptr(nvidiaFD), 0)

			if mmapErrno != 0 {
				*(*int64)(unsafe.Pointer(shared + agentOffResultN)) = -1
				*(*int32)(unsafe.Pointer(shared + agentOffErrno)) = int32(mmapErrno)
			} else {
				// Call the RDMA verbs ioctl.
				ioctlN, ioctlErrno := hostsyscall.RawSyscall(
					unix.SYS_IOCTL,
					uintptr(uverbsFD),
					uintptr(ioctlCmd),
					shared+agentOffBuf)
				*(*int64)(unsafe.Pointer(shared + agentOffResultN)) = int64(ioctlN)
				*(*int32)(unsafe.Pointer(shared + agentOffErrno)) = int32(ioctlErrno)
			}

			// Signal completion: set done=1, wake parent.
			*cmdPtr = agentCmdNone
			*donePtr = 1
			hostsyscall.RawSyscall6(unix.SYS_FUTEX,
				uintptr(unsafe.Pointer(donePtr)),
				1, // FUTEX_WAKE
				1, // wake 1 waiter
				0, 0, 0)

		case agentCmdMunmap:
			gpuVA := *(*uint64)(unsafe.Pointer(shared + agentOffGPUVA))
			length := *(*uint64)(unsafe.Pointer(shared + agentOffLen))
			hostsyscall.RawSyscall(unix.SYS_MUNMAP, uintptr(gpuVA), uintptr(length), 0)

			*cmdPtr = agentCmdNone
			*donePtr = 1
			hostsyscall.RawSyscall6(unix.SYS_FUTEX,
				uintptr(unsafe.Pointer(donePtr)),
				1, 1, 0, 0, 0)

		case agentCmdExit:
			hostsyscall.RawSyscall(unix.SYS_EXIT, 0, 0, 0)
		}
	}
}

// sendCommand sends a command to the agent and waits for completion.
func (a *gpuAgent) sendCommand() {
	donePtr := (*uint32)(unsafe.Pointer(a.sharedPage + agentOffDone))
	cmdPtr := (*uint32)(unsafe.Pointer(a.sharedPage + agentOffCmd))

	// Reset done flag, set command, wake agent.
	atomic.StoreUint32(donePtr, 0)
	// The command type was already written by the caller.
	// Wake the agent's futex_wait on cmdPtr.
	hostsyscall.RawSyscall6(unix.SYS_FUTEX,
		uintptr(unsafe.Pointer(cmdPtr)),
		1, // FUTEX_WAKE
		1, // wake 1 waiter
		0, 0, 0)

	// Wait for completion: futex_wait until *donePtr != 0.
	for {
		done := atomic.LoadUint32(donePtr)
		if done != 0 {
			break
		}
		hostsyscall.RawSyscall6(unix.SYS_FUTEX,
			uintptr(unsafe.Pointer(donePtr)),
			0, // FUTEX_WAIT
			0, // expected value
			0, 0, 0)
	}
}

// munmapVMA asks the agent to munmap a VMA range.
func (a *gpuAgent) munmapVMA(gpuVA, length uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	sp := a.sharedPage
	*(*uint64)(unsafe.Pointer(sp + agentOffGPUVA)) = gpuVA
	*(*uint64)(unsafe.Pointer(sp + agentOffLen)) = length
	atomic.StoreUint32((*uint32)(unsafe.Pointer(sp+agentOffCmd)), agentCmdMunmap)
	a.sendCommand()
}

// beforeFork and afterFork are provided by the Go runtime to make
// fork-like operations safe. They mask signals and serialize with the GC.
//
//go:linkname beforeFork syscall.runtime_BeforeFork
func beforeFork()

//go:linkname afterFork syscall.runtime_AfterFork
func afterFork()

//go:linkname afterForkInChild syscall.runtime_AfterForkInChild
func afterForkInChild()
