LDFLAGS = -fopenmp
NUM_CORES_BENCH ?= NUM_CORES
CFLAGS = -O0 -D IS_MEMPOOL=0 -DNUM_CORES_BENCH=$(NUM_CORES_BENCH) -I../my_argmax_data
CC = gcc

main: main.c ../my_argmax_data/data.h ../my_argmax_data/no_mempool.h
	${CC} ${CFLAGS} ${LDFLAGS} main.c -o main

dump: main
	objdump -d main > main.dump

run: main dump
	#../../../install/llvm/bin/clang-format -i main.c
	./main

clean:
	rm -f main main.dump
