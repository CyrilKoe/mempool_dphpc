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

# Top-K

## Setup

### Data generation

The data generation tool in *topk/utils/datagen* can be used to generate representative data for the top-k kernel. To generate a set of random inputs, run:
```bash
cd topk/utils/datagen
python3 datagen.py all --iterations <n_iteration> --nseeds <n_seeds> --precision <n_bits> --signed <True|False>
```
A set of C header files contataining random inputs for different values of N and K (open datagen.py to see which values or to change them) will be generated in the *headers* sub-directory. For more details about the data generation strategy refer to topk/README.md and the documentation in the datagen.py script.

### Golden Model

A reference implementation of the top-k algorithm can be found in *topk/utils/reference*. This model can be used to generate the expected output of the Mempool's top-k implementation for a specific input data. To run the model, create a header file called *data.h* in the *topk/utils/reference/include* directory containing the input vector. Then run:
```bash
cd topk/utils/reference
make clean all
./build/topk
```
The output of the algorithm will be printed to standard output once the execution has finished.

## Execution

### **Important note**

The implementation of the Top-K kernels makes a few simple assumptions about Mempool's architecture configuration. In particular, the *seq_mem_size* parameter ($MEMPOOL_ROOT/config/config.mk) is required to be set to 2048 (instead of 1024), to guarantee that there is enough space for the dynamic memory allocator. Furthermore, the compilation should be run with the provided makefiles, as explained below.

### Compile and run baseline

To run and compare the top-k baselines on Mempool, navigate to the *topk/baseline* directory and copy one of the generated input header files in the *include* directory. Then, set the C macro *USE_HEAP* in main.c to determine the algorithm to execute. Finally, compile and run the simulation.
```bash
# Compile the source files
cd topk/baseline
make clean all

# Run the hardware simulation
cd $MEMPOOL_ROOT/hardware
preload=$MEMPOOL_ROOT/software/onnx/topk/baseline/build/topk make simcvcs

# Generate traces
make trace
```

### Compile and run parallel top-k

The parallel top-k implementation can be found in *topk/multicore* directory. Use the macros described in topk/README.md to specify the number of cores and the top-k parameters. As for the baseline, copy one of the generated input header files in the *include* directory. Finally, compile and run the simulation.
```bash
# Compile the source files
cd topk/multicore
make clean all

# Run the hardware simulation
cd $MEMPOOL_ROOT/hardware
preload=$MEMPOOL_ROOT/software/onnx/topk/multicore/build/topk make simcvcs

# Generate traces
make trace
```

# Target list

- argmax_base : sequential argmax
- argmax_impl : optimized argmax
- conv1d_8b : set of optimized conv1d kernels
- topk : single- and multi-core optimized top-k

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
