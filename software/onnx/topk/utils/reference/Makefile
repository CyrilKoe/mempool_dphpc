APP=topk
SRCS=heapify.c
INCLUDE_DIR=include
CC=gcc
CCOPT=-O3 -std=c99
BUILD=build

all: $(SRCS)
	mkdir -p $(BUILD)
	$(CC) $(CCOPT) -I$(INCLUDE_DIR) $(SRCS) -o $(BUILD)/$(APP)

clean:
	rm -rf $(BUILD)
