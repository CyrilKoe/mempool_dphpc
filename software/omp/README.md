# my_argmax_dyn_*
You can run them on your local machine with the Makefile given in the /my_argmax_dyn_* folder
You need to modify your `config/mempool.mk` by adding :
```
seq_mem_size ?= 2048
```
First go in the my_argmax_dyn* dir and run `make run` to build the dirty soft links

# OpenMP Applications

> :warning: **The OpenMP runtime and applications are work in progress. Currently, we only support GOMP (GCC).**
