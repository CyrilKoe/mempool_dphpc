LDFLAGS = -fopenmp
CFLAGS = -D IS_MEMPOOL=0
SIZE = 1024
MAX = 100
CC = gcc-11.2.0

data.h:
	../../../scripts/gen_data.py --clangformat=clang-format -o data.h --variable=l2_data --size=$(SIZE) --min=-$(MAX) --max=$(MAX)

main: main.c data.h
	${CC} ${CFLAGS} ${LDFLAGS} main.c -o main

dump: main
	#objdump -d main > main.dump

run: main dump
	clang-format -i main.c
	./main
