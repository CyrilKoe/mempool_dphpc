// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Victor Jung, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "kernel/conv1d_i8.h"

#define INPUT_LENGTH 256
#define INPUT_CHANNELS 1
#define KERNEL_LENGTH 16

#define BATCH_SIZE 1
#define PAD_BEGIN 8
#define PAD_END 8
#define STRIDES 1

#define OUTPUT_LENGTH 256
#define OUTPUT_CHANNELS 1

#define KERNEL_COPY 1

// #define VERBOSE_IN
#define VERBOSE_OUT

#define NUM_TILES (NUM_CORES/NUM_CORES_PER_TILE)

// volatile int8_t in[INPUT_LENGTH * INPUT_CHANNELS + PAD_BEGIN + PAD_END] __attribute__((section(".l1_prio"))); // interleaved
volatile int8_t* in[NUM_TILES] __attribute__((section(".l1_prio"))); // interleaved
volatile int32_t out[OUTPUT_LENGTH * OUTPUT_CHANNELS] __attribute__((section(".l1_prio")));
// volatile int8_t kernel[KERNEL_LENGTH * INPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL_COPY] __attribute__((section(".l1_prio")));
volatile int error __attribute__((section(".l1")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  int8_t kernel[KERNEL_LENGTH*KERNEL_COPY];

  uint32_t corePerKernel = num_cores/KERNEL_COPY;
  
  for(int i = 0; i < KERNEL_COPY; i++){
    kernel[i*KERNEL_LENGTH + 0] = 42;
    kernel[i*KERNEL_LENGTH + 1] = 16;
    kernel[i*KERNEL_LENGTH + 2] = -95;
    kernel[i*KERNEL_LENGTH + 3] = 11;

    kernel[i*KERNEL_LENGTH + 4] = 106;
    kernel[i*KERNEL_LENGTH + 5] = 75;
    kernel[i*KERNEL_LENGTH + 6] = -12;
    kernel[i*KERNEL_LENGTH + 7] = -69;

    kernel[i*KERNEL_LENGTH + 8] = -34;
    kernel[i*KERNEL_LENGTH + 9] = 1;
    kernel[i*KERNEL_LENGTH + 10] = 15;
    kernel[i*KERNEL_LENGTH + 11] = 0;

    kernel[i*KERNEL_LENGTH + 12] = -91;
    kernel[i*KERNEL_LENGTH + 13] = 29;
    kernel[i*KERNEL_LENGTH + 14] = -43;
    kernel[i*KERNEL_LENGTH + 15] = 54;
  }
  
  mempool_init(core_id, num_cores);
  mempool_barrier_init(core_id);
  mempool_barrier(num_cores);

  if (core_id == 0) {

  // Print out linked list of free blocks
  for (int i = 0; i < NUM_TILES; i++) {
    alloc_dump(get_alloc_tile(i));
  }

  for (int i = 0; i < NUM_TILES; i++) {
    in[i] = domain_malloc(get_alloc_tile(i), 1024);
  }

  for (int i = 0; i < NUM_TILES; i++) {
    alloc_dump(get_alloc_tile(i));
  }
    // Initialize error
    error = 0;

    uint32_t corePerKernel2 = num_cores/KERNEL_COPY;

    printf("Core per Kernel: %d\n", corePerKernel2);

    // Initialize Input
    printf("Init Pad\n");
    for(int i = 0; i < PAD_BEGIN + INPUT_CHANNELS*INPUT_LENGTH + PAD_END; i++) {
      in[i] = 0;
    }
    printf("Init Input\n");
    for(int i = PAD_BEGIN; i < INPUT_CHANNELS*INPUT_LENGTH - PAD_END; i++) {
      in[i] = i % 42;
    }

    // Initialize output
    printf("Init Output\n");
    for(int i = 0; i < OUTPUT_LENGTH * OUTPUT_CHANNELS; i++) {
      out[i] = 0;
    }

    printf("Start v1 kernel benchmark\n");
  }

  mempool_barrier(num_cores);
  mempool_start_benchmark();

  if (core_id == 0) {
    num_cores = 1;
    conv1d_i8_v1(in, out, kernel, BATCH_SIZE, INPUT_LENGTH, 
                INPUT_CHANNELS, KERNEL_LENGTH, corePerKernel, OUTPUT_CHANNELS, 
                PAD_BEGIN, PAD_END, STRIDES, core_id, num_cores);
  }

  mempool_stop_benchmark();
  mempool_barrier(num_cores);

  if(core_id == 0){
    // Initialize output
    for(int i = 0; i < OUTPUT_LENGTH * OUTPUT_CHANNELS; i++) {
      out[i] = 0;
    }
    printf("Start v2 kernel benchmark\n");
  }

  mempool_barrier(num_cores);
  mempool_start_benchmark();

  if (core_id == 0) {
    num_cores = 1;
    conv1d_i8_v2(in, out, kernel, BATCH_SIZE, INPUT_LENGTH, 
                INPUT_CHANNELS, KERNEL_LENGTH, corePerKernel, OUTPUT_CHANNELS, 
                PAD_BEGIN, PAD_END, STRIDES, core_id, num_cores);
  }
  
  mempool_stop_benchmark();
  mempool_barrier(num_cores);

  // if(core_id == 0){
  //   // Initialize output
  //   for(int i = 0; i < OUTPUT_LENGTH * OUTPUT_CHANNELS; i++) {
  //     out[i] = 0;
  //   }
  //   printf("Start v1 kernel 1 core benchmark\n");
  // }

  // mempool_barrier(num_cores);
  // mempool_start_benchmark();

  // if (core_id == 0) {
  //   num_cores = 1;
  //   conv1d_i8_v1(in, out, kernel, BATCH_SIZE, INPUT_LENGTH, 
  //               INPUT_CHANNELS, KERNEL_LENGTH, OUTPUT_CHANNELS, 
  //               GROUP, PAD_BEGIN, PAD_END, STRIDES, core_id, num_cores);
  // }
  
  // mempool_stop_benchmark();
  // mempool_barrier(num_cores);

  // if(core_id == 0){
  //   // Initialize output
  //   for(int i = 0; i < OUTPUT_LENGTH * OUTPUT_CHANNELS; i++) {
  //     out[i] = 0;
  //   }
  //   printf("Start v2 kernel 1 core benchmark\n");
  // }

  // mempool_barrier(num_cores);
  // mempool_start_benchmark();

  // if (core_id == 0) {
  //   num_cores = 1;
  //   conv1d_i8_v2(in, out, kernel, BATCH_SIZE, INPUT_LENGTH, 
  //               INPUT_CHANNELS, KERNEL_LENGTH, OUTPUT_CHANNELS, 
  //               GROUP, PAD_BEGIN, PAD_END, STRIDES, core_id, num_cores);
  // }
  
  // mempool_stop_benchmark();
  // mempool_barrier(num_cores);

  if (core_id == 0) {
    #ifdef VERBOSE_OUT
        printf("In addr %d:\n", in);
        for (int j = 0; j < 10; j++) {
          printf("%d ", in[j]);
        }
        printf("\n");

        printf("out addr %d:\n", out);
        for (int i = 0; i < OUTPUT_CHANNELS; i++) {
          printf("Output Channel %d:\n", i);
          for (int j = 0; j < 10; j++) {
            printf("%d ", out[j]);
          }
          printf("\n");
          printf("%d \n", out[255]);
        }
    #endif
  }

  // wait until all cores have finished
  mempool_barrier(num_cores);

  return error;
}
