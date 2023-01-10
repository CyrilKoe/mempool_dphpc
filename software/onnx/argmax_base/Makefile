LDFLAGS = -fopenmp
NUM_CORES_BENCH ?= NUM_CORES
CFLAGS = -O0 -D IS_MEMPOOL=0 -DNUM_CORES_BENCH=$(NUM_CORES_BENCH) -I../argmax_utils -I../commons
CC = gcc

main: main.c ../argmax_utils/data.h ../commons/no_mempool.h
	${CC} ${CFLAGS} ${LDFLAGS} main.c -o main

dump: main
	#objdump -d main > main.dump

run: main dump
	#../../../install/llvm/bin/clang-format -i main.c
	./main

clean:
	rm -f main main.dump
