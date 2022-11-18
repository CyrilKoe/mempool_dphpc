LDFLAGS = -fopenmp
CFLAGS = -D IS_MEMPOOL=0
CC = gcc-11.2.0

main: main.c data.h
	${CC} ${CFLAGS} ${LDFLAGS} main.c -o main

# Dirty soft link to not modify the omp Makefile
data.h:
	ln -s ../my_argmax_data/data.h data.h

no_mempool.h:
	ln -s ../my_argmax_data/no_mempool.h no_mempool.h

dump: main
	#objdump -d main > main.dump

run: main dump
	../../../install/llvm/bin/clang-format -i main.c
	./main

clean:
	rm -f data.h main main.dump no_mempool.h