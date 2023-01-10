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
#include "xpulp/pooling_xpulp.h"

#define DMA_ADDRESS (0x40010000)

#ifndef PARALLEL_UNROLLED_SIMD
#define PARALLEL_UNROLLED_SIMD 1
#endif

// define matrix dimensions
// B = maxpool(A, K, S) with A[MxM], pooling kernel [KxK] and stride S
#define K 16
#define S 4
#define OUTx 64
#define OUTy 64
// for now hard-coded, but should be OUT + K - S
#define Mx ((OUTx - 1) * S + K)
#define My ((OUTx - 1) * S + K)
#define SIZE (Mx * My * sizeof(int8_t)) //(Mx*My*sizeof(int8_t))

int8_t matrix_A[Mx * My] __attribute__((section(".l1_prio")));
// TODO: implement writing back of result in double-buffered fashion
int8_t matrix_B[((int)((Mx - K) / S) + 1) * ((int)((My - K) / S) + 1)]
    __attribute__((section(".l1_prio")));

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
  }

  mempool_barrier(num_cores);
  uint32_t core_div = 1;

  if (PARALLEL_UNROLLED_SIMD) {
    // COLD CACHE
    if (core_id % core_div == 0) {
      mempool_start_benchmark();
      pooling_unrolled_parallel_i8_rv32im(matrix_A, matrix_B, Mx, My, K, S,
                                          core_id, num_cores / core_div);
      mempool_stop_benchmark();
    }

    // all other cores wait until core 0 has finished
    if (core_id == 0) {
      wake_up_all();
    }

    mempool_wfi();

    // HOT CACHE
    if (core_id % core_div == 0) {
      mempool_start_benchmark();
      pooling_unrolled_parallel_i8_rv32im(matrix_A, matrix_B, Mx, My, K, S,
                                          core_id, num_cores / core_div);
      mempool_stop_benchmark();
    }

    // all other cores wait until core 0 has finished
    if (core_id == 0) {
      wake_up_all();
    }

    mempool_wfi();
  }

  // wait until all cores have finished
  mempool_barrier(num_cores);

  return 0;
}