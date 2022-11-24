// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Viviane Potocnik, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "data.h"
#include "dma.h"
#include "encoding.h"
#include "mempool_dma_frontend.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "kernel/pooling.h"

#ifndef UNROLL
#define UNROLL 1
#endif
#ifndef GROUP
#define GROUP 1
#endif

#define DMA_ADDRESS (0x40010000)


// define matrix dimensions
// B = maxpool(A, K, S) with A[MxM], pooling kernel [KxK] and stride S
#define K 2
#define S 1
#define OUT 4
// for now hard-coded, but should be OUT_SIZE + K - S
#define M 5
#define SIZE (M*M*sizeof(int32_t))

int32_t matrix_A[M * M] __attribute__((section(".l1_prio")));
// int32_t matrix_B[((int)((M - K)/S) + 1) * ((int)((M - K)/S) + 1)] __attribute__((section(".l1_prio")));

int volatile error __attribute__((section(".l1")));

int main() {
  // uint32_t num_cores_per_group = NUM_CORES / NUM_GROUPS;
  uint32_t core_id = mempool_get_core_id();
  // uint32_t group_id = core_id / num_cores_per_group;
  uint32_t num_cores = mempool_get_core_count();
  // Initialize barrier and synchronize
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    
    // Copy the input matrix from L2 to L1 memory
    printf("Copying %d bytes into L1 memory\n", SIZE);
    dma_memcpy_blocking(matrix_A, l2_data, SIZE);
    printf("DMA transfer done.\n");

    
    // Benchmark max pooling kernel
    printf("Starting sequential pooling...\n");
    mempool_start_benchmark();
    max_pooling_sequential(matrix_A, M, K, S);
    mempool_stop_benchmark();
    printf("Sequential pooling done...\n");
    printf("Starting parallel pooling...\n");
  }

  // wait until all cores have finished
  mempool_barrier(num_cores);

  mempool_start_benchmark();
  max_pooling_parallel(matrix_A, M, K, S, core_id, num_cores);
  mempool_stop_benchmark();


  // wait until all cores have finished
  mempool_barrier(num_cores);

  return 0;
}