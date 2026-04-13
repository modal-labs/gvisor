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
// The agent is spawned via clone(2) without CLONE_VM, giving it a copy-on-write
// fork of the sentry's address space with its own mm_struct. It communicates
// with the sentry via a shared memory page and pipe pair.
//
// The agent's syscall loop uses only raw syscalls (no Go runtime) because
// the Go runtime is not fork-safe.
type gpuAgent struct {
	devName string
	pid     int32

	// cmdPipe is written by the sentry to wake the agent.
	// resPipe is written by the agent to signal completion.
	cmdPipe int32
	resPipe int32

	// sharedPage is a MAP_SHARED anonymous page for passing ioctl buffers
	// and results between the sentry and agent. Both processes see the
	// same physical page.
	sharedPage uintptr
}

// Shared page layout constants.
const (
	// Command type at offset 0.
	agentCmdMmapAndIoctl = 1
	agentCmdMunmap       = 2
	agentCmdExit         = 3

	// Shared page field offsets.
	agentOffCmd       = 0  // u32: command type
	agentOffGPUVA     = 8  // u64: GPU virtual address for mmap
	agentOffLen       = 16 // u64: length for mmap
	agentOffNvidiaFD  = 24 // i32: nvidia device FD for mmap
	agentOffUverbsFD  = 28 // i32: uverbs host FD for ioctl
	agentOffIoctlCmd  = 32 // u32: ioctl command number
	agentOffResultN   = 40 // i64: ioctl return value
	agentOffErrno     = 48 // i32: ioctl errno
	agentOffBufLen    = 52 // u32: ioctl buffer length
	agentOffBuf       = 64 // ioctl buffer data (up to agentMaxBuf bytes)
	agentMaxBuf       = 4096 - agentOffBuf
	agentSharedSize   = 8192 // two pages to accommodate large buffers
)

// gpuAgents is the global registry of per-GPU agent processes.
var gpuAgents struct {
	mu      sync.Mutex
	byDev   map[string]*gpuAgent
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

	// Create pipe pairs: sentry→agent (cmd), agent→sentry (res).
	var cmdPipe [2]int
	if err := unix.Pipe2(cmdPipe[:], unix.O_CLOEXEC); err != nil {
		unix.RawSyscall(unix.SYS_MUNMAP, sharedPage, agentSharedSize, 0)
		return nil, fmt.Errorf("pipe2 cmd: %v", err)
	}
	var resPipe [2]int
	if err := unix.Pipe2(resPipe[:], unix.O_CLOEXEC); err != nil {
		unix.Close(cmdPipe[0])
		unix.Close(cmdPipe[1])
		unix.RawSyscall(unix.SYS_MUNMAP, sharedPage, agentSharedSize, 0)
		return nil, fmt.Errorf("pipe2 res: %v", err)
	}

	// Clear CLOEXEC on the FDs the child needs.
	clearCloexec(cmdPipe[0]) // child reads commands
	clearCloexec(resPipe[1]) // child writes results

	// Clone without CLONE_VM — child gets its own mm_struct.
	log.Infof("rdmaproxy: about to clone for GPU agent dev=%q", devName)
	beforeFork()
	pid, errno := hostsyscall.RawSyscall(unix.SYS_CLONE, uintptr(unix.SIGCHLD), 0, 0)
	if errno != 0 {
		afterFork()
		unix.Close(cmdPipe[0])
		unix.Close(cmdPipe[1])
		unix.Close(resPipe[0])
		unix.Close(resPipe[1])
		unix.RawSyscall(unix.SYS_MUNMAP, sharedPage, agentSharedSize, 0)
		log.Warningf("rdmaproxy: clone failed for GPU agent dev=%q: errno=%d (%v)", devName, errno, errno)
		return nil, fmt.Errorf("clone: %v", errno)
	}

	if pid == 0 {
		// Child process — raw syscalls only, no Go runtime.
		afterForkInChild()
		gpuAgentLoop(int32(cmdPipe[0]), int32(resPipe[1]), sharedPage)
		// unreachable
	}

	// Parent process.
	afterFork()

	// Close child's ends of pipes.
	unix.Close(cmdPipe[0])
	unix.Close(resPipe[1])

	agent := &gpuAgent{
		devName:    devName,
		pid:        int32(pid),
		cmdPipe:    int32(cmdPipe[1]),
		resPipe:    int32(resPipe[0]),
		sharedPage: sharedPage,
	}
	log.Infof("rdmaproxy: spawned GPU agent pid=%d for dev=%q", pid, devName)
	return agent, nil
}

func clearCloexec(fd int) {
	flags, _ := unix.FcntlInt(uintptr(fd), unix.F_GETFD, 0)
	unix.FcntlInt(uintptr(fd), unix.F_SETFD, flags&^unix.FD_CLOEXEC)
}

// gpuAgentLoop is the child process's main loop. It uses only raw syscalls
// because the Go runtime is not fork-safe.
//
//go:norace
//go:nosplit
func gpuAgentLoop(cmdFD, resFD int32, shared uintptr) {
	var cmdByte [1]byte
	for {
		// Wait for command signal from sentry.
		n, _, errno := hostsyscall.RawSyscall(unix.SYS_READ, uintptr(cmdFD), uintptr(unsafe.Pointer(&cmdByte[0])), 1)
		if n == 0 || errno != 0 {
			hostsyscall.RawSyscall(unix.SYS_EXIT, 1, 0, 0)
		}

		cmd := *(*uint32)(unsafe.Pointer(shared + agentOffCmd))

		switch cmd {
		case agentCmdMmapAndIoctl:
			gpuVA := *(*uint64)(unsafe.Pointer(shared + agentOffGPUVA))
			length := *(*uint64)(unsafe.Pointer(shared + agentOffLen))
			nvidiaFD := *(*int32)(unsafe.Pointer(shared + agentOffNvidiaFD))
			uverbsFD := *(*int32)(unsafe.Pointer(shared + agentOffUverbsFD))
			ioctlCmd := *(*uint32)(unsafe.Pointer(shared + agentOffIoctlCmd))
			bufLen := *(*uint32)(unsafe.Pointer(shared + agentOffBufLen))

			// munmap any existing mapping at the target GPU VA
			// (inherited from parent's COW address space).
			hostsyscall.RawSyscall(unix.SYS_MUNMAP, uintptr(gpuVA), uintptr(length), 0)

			// mmap nvidia FD at the exact GPU VA.
			mapped, _, mmapErrno := hostsyscall.RawSyscall6(unix.SYS_MMAP,
				uintptr(gpuVA), uintptr(length),
				unix.PROT_READ|unix.PROT_WRITE,
				unix.MAP_SHARED|unix.MAP_FIXED,
				uintptr(nvidiaFD), 0)

			if mmapErrno != 0 {
				*(*int64)(unsafe.Pointer(shared + agentOffResultN)) = -1
				*(*int32)(unsafe.Pointer(shared + agentOffErrno)) = int32(mmapErrno)
			} else {
				_ = mapped
				// Call the RDMA verbs ioctl with the buffer from shared page.
				ioctlN, _, ioctlErrno := hostsyscall.RawSyscall(
					unix.SYS_IOCTL,
					uintptr(uverbsFD),
					uintptr(ioctlCmd),
					shared+agentOffBuf)
				*(*int64)(unsafe.Pointer(shared + agentOffResultN)) = int64(ioctlN)
				if ioctlErrno != 0 {
					*(*int32)(unsafe.Pointer(shared + agentOffErrno)) = int32(ioctlErrno)
				} else {
					*(*int32)(unsafe.Pointer(shared + agentOffErrno)) = 0
				}
			}
			_ = bufLen

			// Signal completion.
			resByte := [1]byte{1}
			hostsyscall.RawSyscall(unix.SYS_WRITE, uintptr(resFD), uintptr(unsafe.Pointer(&resByte[0])), 1)

		case agentCmdMunmap:
			gpuVA := *(*uint64)(unsafe.Pointer(shared + agentOffGPUVA))
			length := *(*uint64)(unsafe.Pointer(shared + agentOffLen))
			hostsyscall.RawSyscall(unix.SYS_MUNMAP, uintptr(gpuVA), uintptr(length), 0)

			resByte := [1]byte{1}
			hostsyscall.RawSyscall(unix.SYS_WRITE, uintptr(resFD), uintptr(unsafe.Pointer(&resByte[0])), 1)

		case agentCmdExit:
			hostsyscall.RawSyscall(unix.SYS_EXIT, 0, 0, 0)
		}
	}
}

// munmapVMA asks the agent to munmap a VMA range.
func (a *gpuAgent) munmapVMA(gpuVA, length uint64) {
	sp := a.sharedPage
	*(*uint32)(unsafe.Pointer(sp + agentOffCmd)) = agentCmdMunmap
	*(*uint64)(unsafe.Pointer(sp + agentOffGPUVA)) = gpuVA
	*(*uint64)(unsafe.Pointer(sp + agentOffLen)) = length

	cmdByte := [1]byte{1}
	unix.Write(int(a.cmdPipe), cmdByte[:])

	var resByte [1]byte
	unix.Read(int(a.resPipe), resByte[:])
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

