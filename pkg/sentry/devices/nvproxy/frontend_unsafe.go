// Copyright 2023 The gVisor Authors.
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

package nvproxy

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"unsafe"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/nvgpu"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/marshal/primitive"
)

func dedupeStrings(ss []string) []string {
	seen := make(map[string]struct{}, len(ss))
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func gpuMapDeviceInfo(ctx context.Context, client *rootClient, hDevice nvgpu.Handle, fallback string) (nvgpu.Handle, nvgpu.ClassID, []string) {
	names := []string{fallback}
	if hDevice.Val == nvgpu.NV01_NULL_OBJECT {
		return hDevice, 0, names
	}

	deviceObj := client.getObject(ctx, hDevice)
	if deviceObj != nil && deviceObj.class == nvgpu.NV20_SUBDEVICE_0 && deviceObj.parent.Val != nvgpu.NV01_NULL_OBJECT {
		hDevice = deviceObj.parent
		deviceObj = client.getObject(ctx, hDevice)
	}
	if deviceObj == nil {
		return hDevice, 0, names
	}

	if deviceObj.class == nvgpu.NV01_DEVICE_0 {
		if rmObj, ok := deviceObj.impl.(*rmAllocObject); ok {
			var params nvgpu.NV0080_ALLOC_PARAMETERS
			if len(rmObj.params.allocParams) >= params.SizeBytes() {
				params.UnmarshalBytes(rmObj.params.allocParams)
				return hDevice, deviceObj.class, []string{fmt.Sprintf("nvidia%d", params.DeviceID)}
			}
		}
	}
	return hDevice, deviceObj.class, dedupeStrings(names)
}

type gpuMapTarget struct {
	hDevice      nvgpu.Handle
	hDeviceClass nvgpu.ClassID
	mapDevNames  []string
	source       string
}

func gpuMapTargetKey(target gpuMapTarget) string {
	return fmt.Sprintf("%#x|%#x|%s", target.hDevice.Val, target.hDeviceClass, strings.Join(target.mapDevNames, ","))
}

func (fd *frontendFD) queryCardInfo(ctx context.Context) ([]nvgpu.IoctlCardInfo, error) {
	var cardInfo nvgpu.IoctlCardInfo
	cardInfoSize := cardInfo.SizeBytes()
	buf := make([]byte, nvgpu.NV_MAX_DEVICES*cardInfoSize)
	_, _, errno := unix.Syscall(
		unix.SYS_IOCTL,
		uintptr(fd.hostFD),
		frontendIoctlCmd(nvgpu.NV_ESC_CARD_INFO, uint32(len(buf))),
		uintptr(unsafe.Pointer(&buf[0])),
	)
	runtime.KeepAlive(buf)
	if errno != 0 {
		return nil, errno
	}

	cardInfos := make([]nvgpu.IoctlCardInfo, 0, nvgpu.NV_MAX_DEVICES)
	for i := 0; i < nvgpu.NV_MAX_DEVICES; i++ {
		var info nvgpu.IoctlCardInfo
		info.UnmarshalBytes(buf[i*cardInfoSize : (i+1)*cardInfoSize])
		if info.Valid == 0 {
			continue
		}
		cardInfos = append(cardInfos, info)
	}
	if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: queried %d GPU card infos via hostFD=%d", len(cardInfos), fd.hostFD)
	}
	return cardInfos, nil
}

func (fd *frontendFD) queryGPUUUIDFromGPUID(clientH nvgpu.Handle, gpuID uint32) (string, error) {
	ctrlParams := nvgpu.NV0000_CTRL_GPU_GET_UUID_FROM_GPU_ID_PARAMS{
		GPUID:      gpuID,
		UUIDStrLen: nvgpu.NV0000_GPU_MAX_GID_LENGTH,
	}
	ioctlParams := nvgpu.NVOS54_PARAMETERS{
		HClient:    clientH,
		HObject:    clientH,
		Cmd:        nvgpu.NV0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID,
		Params:     p64FromPtr(unsafe.Pointer(&ctrlParams)),
		ParamsSize: uint32(ctrlParams.SizeBytes()),
	}
	n, _, errno := unix.Syscall(
		unix.SYS_IOCTL,
		uintptr(fd.hostFD),
		frontendIoctlCmd(nvgpu.NV_ESC_RM_CONTROL, nvgpu.SizeofNVOS54Parameters),
		uintptr(unsafe.Pointer(&ioctlParams)),
	)
	runtime.KeepAlive(&ctrlParams)
	runtime.KeepAlive(&ioctlParams)
	if errno != 0 {
		return "", errno
	}
	if ioctlParams.Status != nvgpu.NV_OK {
		return "", fmt.Errorf("status=%#x n=%#x", ioctlParams.Status, n)
	}
	gpuUUID := strings.TrimRight(string(ctrlParams.GPUUUID[:]), "\x00")
	if gpuUUID == "" {
		return "", fmt.Errorf("empty uuid for gpuID=%#x", gpuID)
	}
	return gpuUUID, nil
}

func (fd *frontendFD) clientGPUUUIDToMinor(ctx context.Context, client *rootClient) map[string]uint32 {
	if len(client.gpuUUIDToMinor) != 0 {
		if ctx.IsLogging(log.Debug) {
			ctx.Debugf("nvproxy: reusing %d cached client GPU UUID->minor mappings for client %v via hostFD=%d", len(client.gpuUUIDToMinor), client.handle, fd.hostFD)
		}
		return client.gpuUUIDToMinor
	}

	queryFD := client.params.fd
	if queryFD == nil {
		queryFD = fd
	}
	cardInfos, err := queryFD.queryCardInfo(ctx)
	if err != nil {
		ctx.Debugf("nvproxy: failed to query card info via hostFD=%d: %v", queryFD.hostFD, err)
		return nil
	}
	if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: resolving GPU UUID->minor mappings for client %v via hostFD=%d from %d card infos", client.handle, queryFD.hostFD, len(cardInfos))
	}

	gpuUUIDToMinor := make(map[string]uint32, len(cardInfos))
	for _, info := range cardInfos {
		if ctx.IsLogging(log.Debug) {
			ctx.Debugf("nvproxy: card info candidate for client %v via hostFD=%d: gpuID=%#x minor=%d valid=%d", client.handle, queryFD.hostFD, info.GPUID, info.MinorNumber, info.Valid)
		}
		gpuUUID, err := queryFD.queryGPUUUIDFromGPUID(client.handle, info.GPUID)
		if err != nil {
			ctx.Debugf("nvproxy: failed to query GPU UUID for gpuID=%#x minor=%d via hostFD=%d: %v", info.GPUID, info.MinorNumber, queryFD.hostFD, err)
			continue
		}
		if ctx.IsLogging(log.Debug) {
			ctx.Debugf("nvproxy: resolved GPU UUID %q -> minor=%d for client %v via hostFD=%d", gpuUUID, info.MinorNumber, client.handle, queryFD.hostFD)
		}
		gpuUUIDToMinor[gpuUUID] = info.MinorNumber
	}
	if len(gpuUUIDToMinor) != 0 {
		client.gpuUUIDToMinor = gpuUUIDToMinor
		if ctx.IsLogging(log.Debug) {
			ctx.Debugf("nvproxy: cached %d GPU UUID->minor mappings for client %v via hostFD=%d", len(gpuUUIDToMinor), client.handle, queryFD.hostFD)
		}
	} else if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: resolved no GPU UUID->minor mappings for client %v via hostFD=%d", client.handle, queryFD.hostFD)
	}
	return client.gpuUUIDToMinor
}

func findGPUDeviceByMinor(client *rootClient, minor uint32) (nvgpu.Handle, nvgpu.ClassID, bool) {
	for _, obj := range client.resources {
		if obj.class != nvgpu.NV01_DEVICE_0 {
			continue
		}
		rmObj, ok := obj.impl.(*rmAllocObject)
		if !ok {
			continue
		}
		var params nvgpu.NV0080_ALLOC_PARAMETERS
		if len(rmObj.params.allocParams) < params.SizeBytes() {
			continue
		}
		params.UnmarshalBytes(rmObj.params.allocParams)
		if params.DeviceID == minor {
			return obj.handle, obj.class, true
		}
	}
	return nvgpu.Handle{}, 0, false
}

func (fd *frontendFD) gpuMapTargets(ctx context.Context, client *rootClient, mapping gpuExternalAllocation, memObj *object) []gpuMapTarget {
	targets := make([]gpuMapTarget, 0, len(mapping.gpuUUIDs)+1)
	seen := make(map[string]struct{}, len(mapping.gpuUUIDs)+1)
	addTarget := func(target gpuMapTarget) {
		if target.hDevice.Val == nvgpu.NV01_NULL_OBJECT {
			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: skipping GPU map target with null device for GPU VA %#x source=%s mapDevNames=%v", mapping.base, target.source, target.mapDevNames)
			}
			return
		}
		target.mapDevNames = dedupeStrings(target.mapDevNames)
		if len(target.mapDevNames) == 0 {
			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: skipping GPU map target with no device names for GPU VA %#x source=%s hDevice=%v class=%v", mapping.base, target.source, target.hDevice, target.hDeviceClass)
			}
			return
		}
		key := gpuMapTargetKey(target)
		if _, ok := seen[key]; ok {
			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: skipping duplicate GPU map target for GPU VA %#x source=%s hDevice=%v class=%v mapDevNames=%v", mapping.base, target.source, target.hDevice, target.hDeviceClass, target.mapDevNames)
			}
			return
		}
		seen[key] = struct{}{}
		targets = append(targets, target)
		if ctx.IsLogging(log.Debug) {
			ctx.Debugf("nvproxy: added GPU map target for GPU VA %#x source=%s hDevice=%v class=%v mapDevNames=%v", mapping.base, target.source, target.hDevice, target.hDeviceClass, target.mapDevNames)
		}
	}

	gpuUUIDToMinor := fd.dev.nvp.gpuUUIDToMinor
	uuidSource := "global"
	if len(gpuUUIDToMinor) == 0 {
		gpuUUIDToMinor = fd.clientGPUUUIDToMinor(ctx, client)
		uuidSource = "client"
	}
	if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: selecting GPU map targets for GPU VA %#x len=%d offset=%#x hClient=%v hMemory=%v gpuUUIDs=%v memObjParent=%v uuidSource=%s uuidMapSize=%d",
			mapping.base, mapping.length, mapping.offset, mapping.hClient, mapping.hMemory, mapping.gpuUUIDs, memObj.parent, uuidSource, len(gpuUUIDToMinor))
	}
	for _, gpuUUID := range mapping.gpuUUIDs {
		minor, ok := gpuUUIDToMinor[gpuUUID]
		if !ok {
			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: no minor mapping for GPU UUID %q while preparing GPU VA %#x (uuidSource=%s mapSize=%d)", gpuUUID, mapping.base, uuidSource, len(gpuUUIDToMinor))
			}
			continue
		}
		hDevice, hDeviceClass, ok := findGPUDeviceByMinor(client, minor)
		if !ok {
			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: no device handle found for GPU UUID %q (minor=%d) while preparing GPU VA %#x", gpuUUID, minor, mapping.base)
			}
			continue
		}
		addTarget(gpuMapTarget{
			hDevice:      hDevice,
			hDeviceClass: hDeviceClass,
			mapDevNames:  []string{fmt.Sprintf("nvidia%d", minor)},
			source:       fmt.Sprintf("uvm gpuUUID=%q", gpuUUID),
		})
	}

	if memObj != nil {
		hDevice, hDeviceClass, mapDevNames := gpuMapDeviceInfo(ctx, client, memObj.parent, fd.dev.basename())
		addTarget(gpuMapTarget{
			hDevice:      hDevice,
			hDeviceClass: hDeviceClass,
			mapDevNames:  mapDevNames,
			source:       fmt.Sprintf("memory parent=%v", memObj.parent),
		})
	}
	baseTargets := append([]gpuMapTarget(nil), targets...)
	if len(baseTargets) > 1 {
		for i, deviceTarget := range baseTargets {
			for j, mapTarget := range baseTargets {
				if i == j {
					continue
				}
				addTarget(gpuMapTarget{
					hDevice:      deviceTarget.hDevice,
					hDeviceClass: deviceTarget.hDeviceClass,
					mapDevNames:  append([]string(nil), mapTarget.mapDevNames...),
					source:       fmt.Sprintf("%s + mapDev from %s", deviceTarget.source, mapTarget.source),
				})
			}
		}
	}
	if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: prepared %d GPU map targets for GPU VA %#x hMemory=%v", len(targets), mapping.base, mapping.hMemory)
	}

	return targets
}

type gpuMapFDCandidate struct {
	devName        string
	hostFD         int32
	sandboxMmapLen uint64
	priority       int
}

func (fd *frontendFD) existingGPUMapFDCandidates(mapDevNames []string) []gpuMapFDCandidate {
	devPriority := make(map[string]int, len(mapDevNames))
	for i, name := range mapDevNames {
		if _, ok := devPriority[name]; ok {
			continue
		}
		devPriority[name] = i
	}

	candidates := make([]gpuMapFDCandidate, 0, len(fd.dev.nvp.frontendFDs))
	seenHostFDs := make(map[int32]struct{})
	addCandidate := func(frontend *frontendFD) {
		hostFD := frontend.hostFD
		if hostFD < 0 {
			return
		}
		devName := frontend.dev.basename()
		priority, ok := devPriority[devName]
		if !ok {
			return
		}
		if _, ok := seenHostFDs[hostFD]; ok {
			return
		}
		seenHostFDs[hostFD] = struct{}{}

		frontend.memmapFile.mmapMu.Lock()
		sandboxMmapLen := frontend.memmapFile.mmapLength
		frontend.memmapFile.mmapMu.Unlock()

		candidates = append(candidates, gpuMapFDCandidate{
			devName:        devName,
			hostFD:         hostFD,
			sandboxMmapLen: sandboxMmapLen,
			priority:       priority,
		})
	}

	fd.dev.nvp.fdsMu.Lock()
	if len(fd.registeredDeviceFDs) != 0 {
		for frontend := range fd.registeredDeviceFDs {
			addCandidate(frontend)
		}
	} else if !fd.dev.isCtlDevice() {
		addCandidate(fd)
	} else {
		for frontend := range fd.dev.nvp.frontendFDs {
			addCandidate(frontend)
		}
	}
	fd.dev.nvp.fdsMu.Unlock()

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].priority != candidates[j].priority {
			return candidates[i].priority < candidates[j].priority
		}
		// Prefer frontends that don't already have an app-visible mmap context.
		if (candidates[i].sandboxMmapLen == 0) != (candidates[j].sandboxMmapLen == 0) {
			return candidates[i].sandboxMmapLen == 0
		}
		if candidates[i].sandboxMmapLen != candidates[j].sandboxMmapLen {
			return candidates[i].sandboxMmapLen < candidates[j].sandboxMmapLen
		}
		return candidates[i].hostFD < candidates[j].hostFD
	})
	return candidates
}

func frontendIoctlInvoke[Params any, PtrParams hasStatusPtr[Params]](fi *frontendIoctlState, ioctlParams PtrParams) (uintptr, error) {
	n, err := frontendIoctlInvokeNoStatus(fi, ioctlParams)
	if err == nil && log.IsLogging(log.Debug) {
		if status := ioctlParams.GetStatus(); status != nvgpu.NV_OK {
			fi.ctx.Debugf("nvproxy: frontend ioctl failed: status=%#x", status)
		}
	}
	return n, err
}

func frontendIoctlInvokeNoStatus[Params any](fi *frontendIoctlState, ioctlParams *Params) (uintptr, error) {
	n, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(fi.fd.hostFD), frontendIoctlCmd(fi.nr, fi.ioctlParamsSize), uintptr(unsafe.Pointer(ioctlParams)))
	runtime.KeepAlive(ioctlParams)
	if errno != 0 {
		return n, errno
	}
	return n, nil
}

func (fd *frontendFD) prepareGPUVMA(ctx context.Context, addr, alignedStart, alignedLen, mapAddr uint64) (int32, string, uint64, error) {
	mapping, ok := fd.findGPUExternalAllocation(addr, alignedStart, alignedLen)
	if !ok {
		return -1, fd.dev.basename(), 0, fmt.Errorf("no UVM external allocation tracked for GPU VA %#x", addr)
	}
	if ctx.IsLogging(log.Debug) {
		ctx.Debugf("nvproxy: preparing GPU VMA for addr=%#x alignedStart=%#x len=%d via %q hostFD=%d base=%#x mapLen=%d offset=%#x hClient=%v hMemory=%v gpuUUIDs=%v",
			addr, alignedStart, alignedLen, fd.dev.basename(), fd.hostFD, mapping.base, mapping.length, mapping.offset, mapping.hClient, mapping.hMemory, mapping.gpuUUIDs)
	}

	client, unlock := fd.dev.nvp.getClientWithLock(ctx, mapping.hClient)
	if client == nil {
		return -1, fd.dev.basename(), 0, fmt.Errorf("missing client %v for GPU VA %#x", mapping.hClient, addr)
	}
	memObj := client.getObject(ctx, mapping.hMemory)
	if memObj == nil {
		unlock()
		return -1, fd.dev.basename(), 0, fmt.Errorf("missing memory object %v for GPU VA %#x", mapping.hMemory, addr)
	}
	targets := fd.gpuMapTargets(ctx, client, mapping, memObj)
	unlock()
	if len(targets) == 0 {
		return -1, fd.dev.basename(), 0, fmt.Errorf("no GPU mapping target found for memory object %v GPU VA %#x", mapping.hMemory, addr)
	}

	delta := alignedStart - mapping.base
	pLinAddr := alignedStart
	if mapAddr != 0 {
		pLinAddr = mapAddr
	}
	var errs []string
	tryMapFD := func(target gpuMapTarget, mapFD int32, mapDevName, mapSource string, closeOnFail, dupOnSuccess bool) (int32, string, uint64, error) {
		ioctlParams := nvgpu.IoctlNVOS33ParametersWithFD{
			Params: nvgpu.NVOS33_PARAMETERS{
				HClient:        mapping.hClient,
				HDevice:        target.hDevice,
				HMemory:        mapping.hMemory,
				Offset:         mapping.offset + delta,
				Length:         alignedLen,
				PLinearAddress: nvgpu.P64(pLinAddr),
			},
			FD: mapFD,
		}
		n, _, errno := unix.Syscall(
			unix.SYS_IOCTL,
			uintptr(fd.hostFD),
			frontendIoctlCmd(nvgpu.NV_ESC_RM_MAP_MEMORY, nvgpu.SizeofIoctlNVOS33ParametersWithFD),
			uintptr(unsafe.Pointer(&ioctlParams)),
		)
		runtime.KeepAlive(&ioctlParams)
		if errno != 0 {
			if closeOnFail {
				unix.Close(int(mapFD))
			}
			errs = append(errs, fmt.Sprintf("target=%s mapDev=%q source=%s ctrlHostFD=%d mapFD=%d gpuVA=%#x len=%d: %v", target.source, mapDevName, mapSource, fd.hostFD, mapFD, alignedStart, alignedLen, errno))
			return -1, "", 0, nil
		}
		if ioctlParams.Params.Status != nvgpu.NV_OK {
			if closeOnFail {
				unix.Close(int(mapFD))
			}
			errs = append(errs, fmt.Sprintf("target=%s mapDev=%q source=%s ctrlHostFD=%d mapFD=%d gpuVA=%#x len=%d status=%#x n=%#x hDevice=%v class=%v", target.source, mapDevName, mapSource, fd.hostFD, mapFD, alignedStart, alignedLen, ioctlParams.Params.Status, n, target.hDevice, target.hDeviceClass))
			return -1, "", 0, nil
		}

		returnFD := mapFD
		if dupOnSuccess {
			dupFD, err := unix.Dup(int(mapFD))
			if err != nil {
				errs = append(errs, fmt.Sprintf("mapDev=%q source=%s ctrlHostFD=%d mapFD=%d dup: %v", mapDevName, mapSource, fd.hostFD, mapFD, err))
				return -1, "", 0, nil
			}
			returnFD = int32(dupFD)
		}

		// Save the RM_MAP_MEMORY params so the GPU agent can replay them.
		fd.lastRMMapMu.Lock()
		fd.lastRMMapCtrlFD = fd.hostFD
		fd.lastRMMapCmd = uint32(frontendIoctlCmd(nvgpu.NV_ESC_RM_MAP_MEMORY, nvgpu.SizeofIoctlNVOS33ParametersWithFD))
		paramBytes := (*[64]byte)(unsafe.Pointer(&ioctlParams))
		copy(fd.lastRMMapRaw[:], paramBytes[:])
		fd.lastRMMapLen = int(unsafe.Sizeof(ioctlParams))
		fd.lastRMMapValid = true
		fd.lastRMMapMu.Unlock()

		if ctx.IsLogging(log.Debug) {
			ctx.Debugf("nvproxy: prepared RM_MAP_MEMORY for GPU VA %#x-%#x via ctrlHostFD=%d mapFD=%d mapDev=%q source=%s target=%s hClient=%v hDevice=%v class=%v hMemory=%v offset=%#x len=%d",
				alignedStart, alignedStart+alignedLen, fd.hostFD, returnFD, mapDevName, mapSource, target.source, mapping.hClient, target.hDevice, target.hDeviceClass, mapping.hMemory, mapping.offset+delta, alignedLen)
		}
		return returnFD, mapDevName, alignedLen, nil
	}

	for _, target := range targets {
		for _, candidate := range fd.existingGPUMapFDCandidates(target.mapDevNames) {
			mapSource := fmt.Sprintf("frontend hostFD=%d sandboxMmapLen=%d", candidate.hostFD, candidate.sandboxMmapLen)
			if preparedFD, preparedDevName, preparedLen, err := tryMapFD(target, candidate.hostFD, candidate.devName, mapSource, false /* closeOnFail */, true /* dupOnSuccess */); err != nil || preparedFD >= 0 {
				return preparedFD, preparedDevName, preparedLen, err
			}
		}

		for _, mapDevName := range target.mapDevNames {
			mapFD, _, err := openHostDevFile(ctx, mapDevName, fd.dev.nvp.useDevGofer, unix.O_RDWR)
			if err != nil {
				errs = append(errs, fmt.Sprintf("target=%s open %q: %v", target.source, mapDevName, err))
				continue
			}
			if preparedFD, preparedDevName, preparedLen, err := tryMapFD(target, mapFD, mapDevName, "fresh open", true /* closeOnFail */, false /* dupOnSuccess */); err != nil || preparedFD >= 0 {
				return preparedFD, preparedDevName, preparedLen, err
			}
		}
	}
	return -1, fd.dev.basename(), 0, fmt.Errorf("NV_ESC_RM_MAP_MEMORY failed for GPU VA %#x len %d (%s)", alignedStart, alignedLen, strings.Join(errs, "; "))
}

// prepareGPUVMADryRun finds the correct GPU allocation and device FDs, builds
// the RM_MAP_MEMORY ioctl parameters, but does NOT call the ioctl. Returns
// the raw params so a GPU agent process can call RM_MAP_MEMORY in its own
// process context. This is needed because the nvidia driver tracks mmap
// contexts per-process.
func (fd *frontendFD) prepareGPUVMADryRun(ctx context.Context, addr, alignedStart, alignedLen uint64) (mapFD int32, devName string, ctrlFD int32, rmMapCmd uint32, rmMapParams []byte, err error) {
	mapping, ok := fd.findGPUExternalAllocation(addr, alignedStart, alignedLen)
	if !ok {
		return -1, fd.dev.basename(), 0, 0, nil, fmt.Errorf("no UVM external allocation tracked for GPU VA %#x", addr)
	}

	client, unlock := fd.dev.nvp.getClientWithLock(ctx, mapping.hClient)
	if client == nil {
		return -1, fd.dev.basename(), 0, 0, nil, fmt.Errorf("missing client %v for GPU VA %#x", mapping.hClient, addr)
	}
	memObj := client.getObject(ctx, mapping.hMemory)
	if memObj == nil {
		unlock()
		return -1, fd.dev.basename(), 0, 0, nil, fmt.Errorf("missing memory object %v for GPU VA %#x", mapping.hMemory, addr)
	}
	targets := fd.gpuMapTargets(ctx, client, mapping, memObj)
	unlock()
	if len(targets) == 0 {
		return -1, fd.dev.basename(), 0, 0, nil, fmt.Errorf("no GPU mapping target found for memory object %v GPU VA %#x", mapping.hMemory, addr)
	}

	delta := alignedStart - mapping.base
	ioctlCmd := uint32(frontendIoctlCmd(nvgpu.NV_ESC_RM_MAP_MEMORY, nvgpu.SizeofIoctlNVOS33ParametersWithFD))

	// Find the first viable FD candidate and build the params.
	var errs []string
	for _, target := range targets {
		for _, candidate := range fd.existingGPUMapFDCandidates(target.mapDevNames) {
			ioctlParams := nvgpu.IoctlNVOS33ParametersWithFD{
				Params: nvgpu.NVOS33_PARAMETERS{
					HClient:        mapping.hClient,
					HDevice:        target.hDevice,
					HMemory:        mapping.hMemory,
					Offset:         mapping.offset + delta,
					Length:         alignedLen,
					PLinearAddress: nvgpu.P64(alignedStart),
				},
				FD: candidate.hostFD,
			}
			paramBytes := make([]byte, unsafe.Sizeof(ioctlParams))
			copy(paramBytes, (*[64]byte)(unsafe.Pointer(&ioctlParams))[:len(paramBytes)])

			// Return candidate.hostFD directly (no dup needed — with
			// CLONE_FILES the agent shares the FD table). The agent
			// uses this same FD for both RM_MAP_MEMORY and mmap.
			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: dry-run prepared RM_MAP_MEMORY params for GPU VA %#x-%#x ctrlHostFD=%d mapFD=%d mapDev=%q hClient=%v hDevice=%v hMemory=%v",
					alignedStart, alignedStart+alignedLen, fd.hostFD, candidate.hostFD, candidate.devName, mapping.hClient, target.hDevice, mapping.hMemory)
			}
			return candidate.hostFD, candidate.devName, fd.hostFD, ioctlCmd, paramBytes, nil
		}

		for _, mapDevName := range target.mapDevNames {
			candidateFD, _, oerr := openHostDevFile(ctx, mapDevName, fd.dev.nvp.useDevGofer, unix.O_RDWR)
			if oerr != nil {
				errs = append(errs, fmt.Sprintf("open %q: %v", mapDevName, oerr))
				continue
			}
			ioctlParams := nvgpu.IoctlNVOS33ParametersWithFD{
				Params: nvgpu.NVOS33_PARAMETERS{
					HClient:        mapping.hClient,
					HDevice:        target.hDevice,
					HMemory:        mapping.hMemory,
					Offset:         mapping.offset + delta,
					Length:         alignedLen,
					PLinearAddress: nvgpu.P64(alignedStart),
				},
				FD: candidateFD,
			}
			paramBytes := make([]byte, unsafe.Sizeof(ioctlParams))
			copy(paramBytes, (*[64]byte)(unsafe.Pointer(&ioctlParams))[:len(paramBytes)])

			if ctx.IsLogging(log.Debug) {
				ctx.Debugf("nvproxy: dry-run prepared RM_MAP_MEMORY params (fresh open) for GPU VA %#x-%#x ctrlHostFD=%d mapFD=%d mapDev=%q hClient=%v hDevice=%v hMemory=%v",
					alignedStart, alignedStart+alignedLen, fd.hostFD, candidateFD, mapDevName, mapping.hClient, target.hDevice, mapping.hMemory)
			}
			return candidateFD, mapDevName, fd.hostFD, ioctlCmd, paramBytes, nil
		}
	}
	return -1, fd.dev.basename(), 0, 0, nil, fmt.Errorf("no viable FD candidate for GPU VA %#x len %d (%s)", alignedStart, alignedLen, strings.Join(errs, "; "))
}

func rmControlInvoke[Params any](fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, ctrlParams *Params) (uintptr, error) {
	defer runtime.KeepAlive(ctrlParams) // since we convert to non-pointer-typed P64
	origParams := ioctlParams.Params
	ioctlParams.Params = p64FromPtr(unsafe.Pointer(ctrlParams))
	n, err := frontendIoctlInvoke(fi, ioctlParams)
	ioctlParams.Params = origParams
	if err != nil {
		return n, err
	}
	if _, err := ioctlParams.CopyOut(fi.t, fi.ioctlParamsAddr); err != nil {
		return n, err
	}
	return n, nil
}

func ctrlClientSystemGetBuildVersionInvoke(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, ctrlParams *nvgpu.NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_PARAMS, driverVersionBuf, versionBuf, titleBuf *byte) (uintptr, error) {
	// *Buf arguments don't need runtime.KeepAlive() since our caller
	// ctrlClientSystemGetBuildVersion() copies them out, keeping them alive
	// during this function.
	origPDriverVersionBuffer := ctrlParams.PDriverVersionBuffer
	origPVersionBuffer := ctrlParams.PVersionBuffer
	origPTitleBuffer := ctrlParams.PTitleBuffer
	ctrlParams.PDriverVersionBuffer = p64FromPtr(unsafe.Pointer(driverVersionBuf))
	ctrlParams.PVersionBuffer = p64FromPtr(unsafe.Pointer(versionBuf))
	ctrlParams.PTitleBuffer = p64FromPtr(unsafe.Pointer(titleBuf))
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.PDriverVersionBuffer = origPDriverVersionBuffer
	ctrlParams.PVersionBuffer = origPVersionBuffer
	ctrlParams.PTitleBuffer = origPTitleBuffer
	if err != nil {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}
	return n, nil
}

func ctrlIoctlHasInfoList[Params any, PtrParams hasCtrlInfoListPtr[Params]](fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParamsValue Params
	ctrlParams := PtrParams(&ctrlParamsValue)

	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}
	var infoList []byte
	if listSize := ctrlParams.ListSize(); listSize > 0 {
		if !rmapiParamsSizeCheck(listSize, nvgpu.CtrlXxxInfoSize) {
			return 0, frontendFailWithStatus(fi, ioctlParams, nvgpu.NV_ERR_INVALID_ARGUMENT)
		}
		infoList = make([]byte, listSize*nvgpu.CtrlXxxInfoSize)
		if _, err := fi.t.CopyInBytes(addrFromP64(ctrlParams.CtrlInfoList()), infoList); err != nil {
			return 0, err
		}
	}

	origInfoList := ctrlParams.CtrlInfoList()
	if infoList == nil {
		ctrlParams.SetCtrlInfoList(p64FromPtr(unsafe.Pointer(nil)))
	} else {
		ctrlParams.SetCtrlInfoList(p64FromPtr(unsafe.Pointer(&infoList[0])))
	}
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.SetCtrlInfoList(origInfoList)
	if err != nil {
		return n, err
	}

	if infoList != nil {
		if _, err := fi.t.CopyOutBytes(addrFromP64(origInfoList), infoList); err != nil {
			return n, err
		}
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}

	return n, nil
}

func ctrlGetNvU32ListInvoke(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, ctrlParams *nvgpu.RmapiParamNvU32List, list []uint32) (uintptr, error) {
	origList := ctrlParams.List
	ctrlParams.List = p64FromPtr(unsafe.Pointer(&list[0]))
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.List = origList
	if err != nil {
		return n, err
	}
	if _, err := primitive.CopyUint32SliceOut(fi.t, addrFromP64(ctrlParams.List), list); err != nil {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}
	return n, nil
}

func ctrlDevGRGetCapsInvoke(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS, ctrlParams *nvgpu.NV0080_CTRL_GET_CAPS_PARAMS, capsTbl []byte) (uintptr, error) {
	origCapsTbl := ctrlParams.CapsTbl
	ctrlParams.CapsTbl = p64FromPtr(unsafe.Pointer(&capsTbl[0]))
	n, err := rmControlInvoke(fi, ioctlParams, ctrlParams)
	ctrlParams.CapsTbl = origCapsTbl
	if err != nil {
		return n, err
	}
	if _, err := primitive.CopyByteSliceOut(fi.t, addrFromP64(ctrlParams.CapsTbl), capsTbl); err != nil {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}
	return n, nil
}

func ctrlDevFIFOGetChannelList(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV0080_CTRL_FIFO_GET_CHANNELLIST_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}
	if !rmapiParamsSizeCheck(ctrlParams.NumChannels, 4 /* sizeof(NvU32) */) {
		// Compare
		// src/nvidia/src/kernel/gpu/fifo/kernel_fifo_ctrl.c:deviceCtrlCmdFifoGetChannelList_IMPL().
		return 0, linuxerr.EINVAL
	}
	channelHandleList := make([]uint32, ctrlParams.NumChannels)
	if _, err := primitive.CopyUint32SliceIn(fi.t, addrFromP64(ctrlParams.PChannelHandleList), channelHandleList); err != nil {
		return 0, err
	}
	channelList := make([]uint32, ctrlParams.NumChannels)
	if _, err := primitive.CopyUint32SliceIn(fi.t, addrFromP64(ctrlParams.PChannelList), channelList); err != nil {
		return 0, err
	}

	origPChannelHandleList := ctrlParams.PChannelHandleList
	origPChannelList := ctrlParams.PChannelList
	ctrlParams.PChannelHandleList = p64FromPtr(unsafe.Pointer(&channelHandleList[0]))
	ctrlParams.PChannelList = p64FromPtr(unsafe.Pointer(&channelList[0]))
	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	ctrlParams.PChannelHandleList = origPChannelHandleList
	ctrlParams.PChannelList = origPChannelList
	if err != nil {
		return n, err
	}

	if _, err := primitive.CopyUint32SliceOut(fi.t, addrFromP64(origPChannelHandleList), channelHandleList); err != nil {
		return n, err
	}
	if _, err := primitive.CopyUint32SliceOut(fi.t, addrFromP64(origPChannelList), channelList); err != nil {
		return n, err
	}
	if _, err := ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return n, err
	}

	return n, nil
}

func ctrlClientSystemGetP2PCapsInitializeArray(origArr nvgpu.P64, gpuCount uint32) (nvgpu.P64, []uint32, bool) {
	// The driver doesn't try and copy memory if the array is null. See
	// src/nvidia/src/kernel/rmapi/embedded_param_copy.c::embeddedParamCopyIn(),
	// case NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS.
	if origArr == 0 {
		return 0, nil, true
	}

	// Params size is gpuCount * gpuCount * sizeof(NvU32).
	// Use uint64 to handle overflow. In the driver, this is handled by
	// portSafeMulU32(). See
	// src/nvidia/src/kernel/rmapi/embedded_param_copy.c::embeddedParamCopyIn().
	numEntries := uint64(gpuCount) * uint64(gpuCount)
	if numEntries == 0 || numEntries*4 > nvgpu.RMAPI_PARAM_COPY_MAX_PARAMS_SIZE {
		return 0, nil, false
	}

	arr := make([]uint32, numEntries)
	return p64FromPtr(unsafe.Pointer(&arr[0])), arr, true
}

func ctrlClientSystemGetP2PCaps(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}

	origBusPeerIDs := ctrlParams.BusPeerIDs
	busPeerIDs, busPeerIDsBuf, ok := ctrlClientSystemGetP2PCapsInitializeArray(origBusPeerIDs, ctrlParams.GpuCount)
	if !ok {
		return 0, frontendFailWithStatus(fi, ioctlParams, nvgpu.NV_ERR_INVALID_ARGUMENT)
	}
	ctrlParams.BusPeerIDs = busPeerIDs

	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	ctrlParams.BusPeerIDs = origBusPeerIDs
	if err != nil {
		return n, err
	}

	if _, err := primitive.CopyUint32SliceOut(fi.t, addrFromP64(origBusPeerIDs), busPeerIDsBuf); err != nil {
		return n, err
	}

	_, err = ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params))
	return n, err
}

func ctrlClientSystemGetP2PCapsV550(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS54_PARAMETERS) (uintptr, error) {
	var ctrlParams nvgpu.NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_V550
	if ctrlParams.SizeBytes() != int(ioctlParams.ParamsSize) {
		return 0, linuxerr.EINVAL
	}
	if _, err := ctrlParams.CopyIn(fi.t, addrFromP64(ioctlParams.Params)); err != nil {
		return 0, err
	}

	origBusPeerIDs := ctrlParams.BusPeerIDs
	busPeerIDs, busPeerIDsBuf, ok := ctrlClientSystemGetP2PCapsInitializeArray(origBusPeerIDs, ctrlParams.GpuCount)
	if !ok {
		return 0, frontendFailWithStatus(fi, ioctlParams, nvgpu.NV_ERR_INVALID_ARGUMENT)
	}
	ctrlParams.BusPeerIDs = busPeerIDs

	origBusEgmPeerIDs := ctrlParams.BusEgmPeerIDs
	busEgmPeerIDs, busEgmPeerIDsBuf, ok := ctrlClientSystemGetP2PCapsInitializeArray(origBusEgmPeerIDs, ctrlParams.GpuCount)
	if !ok {
		return 0, frontendFailWithStatus(fi, ioctlParams, nvgpu.NV_ERR_INVALID_ARGUMENT)
	}
	ctrlParams.BusEgmPeerIDs = busEgmPeerIDs

	n, err := rmControlInvoke(fi, ioctlParams, &ctrlParams)
	ctrlParams.BusPeerIDs = origBusPeerIDs
	ctrlParams.BusEgmPeerIDs = origBusEgmPeerIDs
	if err != nil {
		return n, err
	}

	// If origBufPeerIDS or origBusEgmPeerIDs is null, the corresponding buffer will be nil
	// and CopyUint32SliceOut() will be a no-op.
	if _, err := primitive.CopyUint32SliceOut(fi.t, addrFromP64(origBusPeerIDs), busPeerIDsBuf); err != nil {
		return n, err
	}
	if _, err := primitive.CopyUint32SliceOut(fi.t, addrFromP64(origBusEgmPeerIDs), busEgmPeerIDsBuf); err != nil {
		return n, err
	}
	_, err = ctrlParams.CopyOut(fi.t, addrFromP64(ioctlParams.Params))
	return n, err
}

func rmAllocInvoke[Params any](fi *frontendIoctlState, ioctlParams *nvgpu.NVOS64_PARAMETERS, allocParams *Params, isNVOS64 bool, addObjLocked func(fi *frontendIoctlState, client *rootClient, ioctlParams *nvgpu.NVOS64_PARAMETERS, rightsRequested nvgpu.RS_ACCESS_MASK, allocParams *Params)) (uintptr, error) {
	defer runtime.KeepAlive(allocParams) // since we convert to non-pointer-typed P64

	// Temporarily replace application pointers with sentry pointers.
	origPAllocParms := ioctlParams.PAllocParms
	origPRightsRequested := ioctlParams.PRightsRequested
	var rightsRequested nvgpu.RS_ACCESS_MASK
	if ioctlParams.PRightsRequested != 0 {
		if _, err := rightsRequested.CopyIn(fi.t, addrFromP64(ioctlParams.PRightsRequested)); err != nil {
			return 0, err
		}
		ioctlParams.PRightsRequested = p64FromPtr(unsafe.Pointer(&rightsRequested))
	}
	ioctlParams.PAllocParms = p64FromPtr(unsafe.Pointer(allocParams))

	var (
		client *rootClient
		unlock = func() {}
	)
	if !ioctlParams.HClass.IsRootClient() {
		client, unlock = fi.fd.dev.nvp.getClientWithLock(fi.ctx, ioctlParams.HRoot)
		if client == nil {
			return 0, frontendFailWithStatus(fi, ioctlParams, nvgpu.NV_ERR_INVALID_CLIENT)
		}
	}
	// Invoke the driver ioctl and restore application pointers. We always pass
	// NVOS64Parameters to the driver even if !isNVOS64, as this is handled
	// identically to the equivalent NVOS21Parameters; compare
	// src/nvidia/src/kernel/rmapi/entry_points.c:_nv04AllocWithSecInfo() and
	// _nv04AllocWithAccessSecInfo().
	origParamsSize := fi.ioctlParamsSize
	fi.ioctlParamsSize = nvgpu.SizeofNVOS64Parameters
	n, err := frontendIoctlInvoke(fi, ioctlParams)
	fi.ioctlParamsSize = origParamsSize
	if err == nil && ioctlParams.Status == nvgpu.NV_OK {
		addObjLocked(fi, client, ioctlParams, rightsRequested, allocParams)
	}
	unlock()
	ioctlParams.PAllocParms = origPAllocParms
	ioctlParams.PRightsRequested = origPRightsRequested
	if err != nil {
		return n, err
	}

	// Copy updated params out to the application.
	outIoctlParams := nvgpu.GetRmAllocParamObj(isNVOS64)
	outIoctlParams.FromOS64(*ioctlParams)
	if ioctlParams.PRightsRequested != 0 {
		if _, err := rightsRequested.CopyOut(fi.t, addrFromP64(ioctlParams.PRightsRequested)); err != nil {
			return n, err
		}
	}
	if _, err := outIoctlParams.CopyOut(fi.t, fi.ioctlParamsAddr); err != nil {
		return n, err
	}
	return n, nil
}

func rmIdleChannelsInvoke(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS30_PARAMETERS, clientsBuf, devicesBuf, channelsBuf *byte) (uintptr, error) {
	origClients := ioctlParams.Clients
	origDevices := ioctlParams.Devices
	origChannels := ioctlParams.Channels
	ioctlParams.Clients = p64FromPtr(unsafe.Pointer(clientsBuf))
	ioctlParams.Devices = p64FromPtr(unsafe.Pointer(devicesBuf))
	ioctlParams.Channels = p64FromPtr(unsafe.Pointer(channelsBuf))
	n, err := frontendIoctlInvoke(fi, ioctlParams)
	ioctlParams.Clients = origClients
	ioctlParams.Devices = origDevices
	ioctlParams.Channels = origChannels
	if err != nil {
		return n, err
	}
	if _, err := ioctlParams.CopyOut(fi.t, fi.ioctlParamsAddr); err != nil {
		return n, err
	}
	return n, nil
}

func rmVidHeapControlAllocSize(fi *frontendIoctlState, ioctlParams *nvgpu.NVOS32_PARAMETERS) (uintptr, error) {
	allocSizeParams := (*nvgpu.NVOS32AllocSize)(unsafe.Pointer(&ioctlParams.Data))
	origAddress := allocSizeParams.Address
	var addr uint64
	if allocSizeParams.Address != 0 {
		if _, err := primitive.CopyUint64In(fi.t, addrFromP64(allocSizeParams.Address), &addr); err != nil {
			return 0, err
		}
		allocSizeParams.Address = p64FromPtr(unsafe.Pointer(&addr))
	}

	client, unlock := fi.fd.dev.nvp.getClientWithLock(fi.ctx, ioctlParams.HRoot)
	if client == nil {
		return 0, frontendFailWithStatus(fi, ioctlParams, nvgpu.NV_ERR_INVALID_CLIENT)
	}
	n, err := frontendIoctlInvoke(fi, ioctlParams)
	if err == nil && ioctlParams.Status == nvgpu.NV_OK {
		// src/nvidia/interface/deprecated/rmapi_deprecated_vidheapctrl.c:_rmVidHeapControlAllocCommon()
		if allocSizeParams.Flags&nvgpu.NVOS32_ALLOC_FLAGS_VIRTUAL != 0 {
			// src/nvidia/src/kernel/mem_mgr/virtual_mem.c:virtmemConstruct_IMPL() => refAddDependant()
			fi.fd.dev.nvp.objAdd(fi.ctx, client, allocSizeParams.HMemory, nvgpu.NV50_MEMORY_VIRTUAL, &miscObject{}, ioctlParams.HObjectParent, ioctlParams.HVASpace)
		} else {
			classID := nvgpu.ClassID(nvgpu.NV01_MEMORY_SYSTEM)
			if (allocSizeParams.Attr2>>nvgpu.NVOS32_ATTR2_USE_EGM_SHIFT)&nvgpu.NVOS32_ATTR2_USE_EGM_MASK == nvgpu.NVOS32_ATTR2_USE_EGM_TRUE {
				classID = nvgpu.NV_MEMORY_EXTENDED_USER
			} else if (allocSizeParams.Attr>>nvgpu.NVOS32_ATTR_LOCATION_SHIFT)&nvgpu.NVOS32_ATTR_LOCATION_MASK == nvgpu.NVOS32_ATTR_LOCATION_VIDMEM {
				classID = nvgpu.NV01_MEMORY_LOCAL_USER
			}
			fi.fd.dev.nvp.objAdd(fi.ctx, client, allocSizeParams.HMemory, classID, &miscObject{}, ioctlParams.HObjectParent)
		}
	}
	unlock()
	allocSizeParams.Address = origAddress
	if err != nil {
		return n, err
	}

	if allocSizeParams.Address != 0 {
		if _, err := primitive.CopyUint64Out(fi.t, addrFromP64(allocSizeParams.Address), addr); err != nil {
			return n, err
		}
	}
	if _, err := ioctlParams.CopyOut(fi.t, fi.ioctlParamsAddr); err != nil {
		return n, err
	}

	return n, nil
}
