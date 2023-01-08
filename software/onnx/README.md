# Argmax

## Generate data
Note that some default data are already furnished, generating is optional.
```bash
# MAX : Generate data in Uniform([-MAX, MAX])
# DATA_GEN_SIZE : Generate 256*DATA_GEN_SIZE datas

(cd argmax_utils && make DATA_GEN_SIZE=64 DATA_GEN_SEED=1000 MAX=10 data.h -B)
```
## Run with x86
```bash
cd argmax_base
make run -B
```
## Compile and run for mempool
```bash
# If you want to default make everything
make
# If you want to set a number of cores do this instead
make NUM_CORES_BENCH=16 argmax_impl

# Now go and run
(cd ../../hardware/ && app=onnx/argmax_base make simcvcs)
# More infos about running are available in the hardware README
```

# Target lists

- argmax_base : sequential argmax
- argmax_impl : optimized argmax

# Add your new operator
Add your `main.c` in a folder here (ex `argmax_base`, `argmax_impl`), please start with `operator_`.

If you need particular macros/includes add your `CGLAGS` in `Makefile` as below :
```makefile
# Argmax CFLAGS
my_argmax_cflags:
	$(eval RISCV_CCFLAGS=${RISCV_CCFLAGS} -I$(SOFTWARE_DIR)/onnx/argmax_utils -DNUM_CORES_BENCH=$(NUM_CORES_BENCH))
# Argmax executables
$(BIN_DIR)/$(APP_PREFIX)argmax*: my_argmax_cflags
```