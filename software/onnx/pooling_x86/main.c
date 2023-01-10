// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Viviane Potocnik, ETH Zurich

#include <stdint.h>
#include <string.h>

#ifndef IS_MEMPOOL
#define IS_MEMPOOL 0
#endif

#if IS_MEMPOOL
#include "alloc.h"
#include "dma.h"
#include "encoding.h"
#include "libgomp.h"
#include "mempool_dma_frontend.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#else
#include "../commons/no_mempool.h"
#endif

#include "data.h"
// #include "encoding.h"
// #include "printf.h"
// #include "runtime.h"
// #include "synchronization.h"
#include "kernel/pooling.h"
#define DMA_ADDRESS (0x40010000)

// #define SEQUENTIAL 0
// #define PARALLEL 1
#ifndef SEQUENTIAL
#define SEQUENTIAL 1
#define PARALLEL 0
#else
#define SEQUENTIAL 0
#define PARALLEL 1
#endif

// define matrix dimensions
// B = maxpool(A, K, S) with A[MxM], pooling kernel [KxK] and stride S
// #define K 16
#ifndef K
#define K 16
#endif
#ifndef S
#define S 4
#endif
#ifndef OUTx
#define OUTx 64
#endif
#ifndef OUTy
#define OUTy 64
#endif
// for now hard-coded, but should be OUT + K - S
#define Mx ((OUTx - 1) * S + K)
#define My ((OUTx - 1) * S + K)
#define SIZE (Mx * My * sizeof(int32_t)) //(Mx*My*sizeof(int32_t))

// dump(marker, 3);

int32_t matrix_A[Mx * My] __attribute__((section(".l1_prio")));
// TODO: implement writing back of result in double-buffered fashion
int32_t matrix_B[((int)((Mx - K) / S) + 1) * ((int)((My - K) / S) + 1)]
    __attribute__((section(".l1_prio")));

int volatile error __attribute__((section(".l1")));

int main() {
  // uint32_t num_cores_per_group = NUM_CORES / NUM_GROUPS;
  // uint32_t core_id = mempool_get_core_id();
  // uint32_t group_id = core_id / num_cores_per_group;
  // uint32_t num_cores = mempool_get_core_count();
  // Initialize barrier and synchronize
  // mempool_barrier_init(core_id);

  // if (core_id == 0) {

  // Copy the input matrix from L2 to L1 memory
  // printf("Copying %d bytes into L1 memory\n", SIZE);
  dma_memcpy_blocking(matrix_A, l2_data, SIZE);
  // printf("DMA transfer done.\n");

  if (SEQUENTIAL) {
    // Benchmark max pooling kernel
    // printf("Starting sequential pooling...\n");

    // COLD CACHE
    mempool_start_benchmark();
    printf("Starting sequential pooling...\n");
    for (int i = 0; i < 10000; i++) {
      max_pooling_sequential(matrix_A, matrix_B, My, K, S);
    }
    printf("Sequential pooling done...\n");
    mempool_stop_benchmark();

    // HOT CACHE
    mempool_start_benchmark();
    max_pooling_sequential(matrix_A, matrix_B, My, K, S);
    mempool_stop_benchmark();

    // printf("Sequential pooling done...\n");
  }

  print_benchmark();

  return 0;
}