# Debugging a hang

Check if `py-spy` installed. If not, use gdb, and ask for thread 1 explicitly:
the other threads are oneCCL and SYCL workers and will show you nothing. You must have access to a compute node first.

```bash
pgrep -fa main.py                                   # find the ranks
gdb -p <pid> -batch -ex 'thread 1' -ex 'bt 25'      # per rank
```

Read the stacks across ranks rather than one in isolation:

- **all ranks blocked in the same recv** -> deadlock, look at op ordering
- **one rank elsewhere, the rest in a wait** -> straggler, look at load balance
- **all ranks in `MPI_Comm_dup`** -> communicator exhaustion

Reproduce under `timeout -s KILL` so a hung job does not sit on the allocation,
and write output to the project filesystem rather than piping it over `ssh`,
which buffers.
