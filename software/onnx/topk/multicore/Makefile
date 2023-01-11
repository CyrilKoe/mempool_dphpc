APPNAME=topk
SRCS=lock.c.o heap.c.o topk.c.o main.c.o
BUILD_DIR=build

ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SOFTWARE_DIR := $(abspath $(ROOT_DIR)/../../..)
RUNTIME_DIR := $(abspath $(SOFTWARE_DIR)/runtime)

OUT_DIR := $(abspath $(ROOT_DIR)/$(BUILD_DIR))

include $(RUNTIME_DIR)/runtime.mk

.PHONY: all
all: $(SRCS) $(RUNTIME) $(LINKER_SCRIPT)
	mkdir -p $(OUT_DIR)
	$(RISCV_CC) -Iinclude -Isources/include $(RISCV_LDFLAGS) -o $(OUT_DIR)/$(APPNAME) $(SRCS) $(RUNTIME) -T$(RUNTIME_DIR)/link.ld
	$(RISCV_OBJDUMP) $(RISCV_OBJDUMP_FLAGS) -D $(OUT_DIR)/$(APPNAME) > $(OUT_DIR)/$(APPNAME).dump
	rm $(SRCS)

clean:
	rm -rf ./build
