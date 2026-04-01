Open these first:

- `runsc-rdma-rank0-bench.txt`
  Multinode `runsc-rdma` rank-0 NCCL output and bandwidth table.

- `runc-rank0-bench.txt`
  Matching multinode `runc` rank-0 NCCL output for direct comparison.

- `strace/`
  Multi-threaded `strace -ttT -ff` capture from the `runsc-rdma` boot process.
  Good starting files are:
  - `strace.runsc.3955573`
  - `strace.runsc.3955624`
  - `strace.runsc.3955656`

- `steady-state-strace-1775055892/`
  Steady-state `strace -ttT -ff` capture from the longer `ITERS=200` run.
  High-level takeaway from that window:
  - `futex`: 506939 calls, 45.022861s total
  - `epoll_pwait`: 7600 calls, 15.045460s total
  - `nanosleep`: 22333 calls, 7.653128s total
  - `ioctl`: 8 calls, 0.002735s total

Other files:

- `runsc-rdma-rank1-status.json`
  Node B job status/output snapshot for the successful `runsc-rdma` run.

- `runc-rank1-status.json`
  Node B job status/output snapshot for the successful `runc` run.
