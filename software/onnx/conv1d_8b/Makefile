NUM_CORES_BENCH ?= NUM_CORES
CFLAGS = -O0 -D -DNUM_CORES_BENCH=$(NUM_CORES_BENCH) -I./conv1d_i8.h
CC = gcc

main: main.c ./conv1d_i8.h
	${CC} ${CFLAGS} ${LDFLAGS} main.c -o main

dump: main
	#objdump -d main > main.dump

run: main dump
	#../../../install/llvm/bin/clang-format -i main.c
	./main

clean:
	rm -f main main.dump