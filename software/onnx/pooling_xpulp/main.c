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
#include "xpulp/pooling_xpulp.h"

#define DMA_ADDRESS (0x40010000)

#define PARALLEL_UNROLLED_SIMD 1


// define matrix dimensions
// B = maxpool(A, K, S) with A[MxM], pooling kernel [KxK] and stride S
#define K 16
#define S 4
#define OUTx 16
#define OUTy 16
// for now hard-coded, but should be OUT + K - S
#define Mx ((OUTx - 1) * S + K) 
#define My ((OUTx - 1) * S + K) 
#define SIZE (Mx*My*sizeof(int32_t)) //(Mx*My*sizeof(int32_t))

int32_t matrix_A[Mx * My] __attribute__((section(".l1_prio")));
// TODO: implement writing back of result in double-buffered fashion
int32_t matrix_B[((int)((Mx - K)/S) + 1) * ((int)((My - K)/S) + 1)] __attribute__((section(".l1_prio")));

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

  if (PARALLEL_UNROLLED_SIMD) {
    mempool_start_benchmark();
    pooling_unrolled_parallel_i8_rv32im(matrix_A, matrix_B, Mx, My, K, S, core_id, num_cores);
    mempool_stop_benchmark();
  }


  // wait until all cores have finished
  mempool_barrier(num_cores);

  return 0;
}