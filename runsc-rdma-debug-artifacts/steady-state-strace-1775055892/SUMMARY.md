Steady-state strace capture from the longer runsc-rdma NCCL run.

Capture window: 8 seconds
Source: /tmp/runsc-long-strace-1775055892

Top syscalls by total observed time:
- futex: 506939 calls, 45.022861s total, 88.8us avg
- epoll_pwait: 7600 calls, 15.045460s total, 1979.7us avg
- nanosleep: 22333 calls, 7.653128s total, 342.7us avg
- ppoll: 5524 calls, 0.053523s total, 9.7us avg
- ioctl: 8 calls, 0.002735s total, 341.9us avg

Takeaway:
Observed steady-state time is dominated by futex/epoll/nanosleep waits, not ioctl churn.
