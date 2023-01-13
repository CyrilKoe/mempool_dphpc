NUM_CORES ?= 256

LDFLAGS = -fopenmp
CFLAGS = -DIS_MEMPOOL=0 -DNUM_CORES_BENCH=$(NUM_CORES) -I../argmax_utils -I../commons -g -Og
CC = gcc-11.2.0

main: main.c ../argmax_utils/data.h ../commons/no_mempool.h
	${CC} ${CFLAGS} ${LDFLAGS} main.c -o main

dump: main
	#objdump -d main > main.dump

run: main dump
	../../../install/llvm/bin/clang-format -i main.c
	./main

clean:
	rm -f main main.dump