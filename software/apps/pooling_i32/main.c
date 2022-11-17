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

// Size in words
#ifndef SIZE
#define SIZE ((NUM_CORES) * (NUM_CORES)*2)
#endif
// Assume banking factor of 4

uint32_t l1_data[SIZE] __attribute__((section(".l1_prio")))
__attribute__((aligned(NUM_CORES * 4 * 4)));

// define matrix dimensions
// B = maxpool(A, K, S) with A[MxM], pooling kernel [KxK] and stride S
#define M 256 // for now hard-coded, but should be SQRT(SIZE)

int main() {
  // uint32_t num_cores_per_group = NUM_CORES / NUM_GROUPS;
  uint32_t core_id = mempool_get_core_id();
  // uint32_t group_id = core_id / num_cores_per_group;
  uint32_t num_cores = mempool_get_core_count();
  // Initialize barrier and synchronize
  mempool_barrier_init(core_id);

  if (core_id == 0) {
    
    // Copy the input matrix from L2 to L1 memory
    dma_memcpy_blocking(l1_data, l2_data, SIZE * sizeof(uint32_t));
    

    // Benchmark max pooling kernel
    mempool_start_benchmark();
    mempool_stop_benchmark();

  // wait until all cores have finished
  mempool_barrier(num_cores);

  return 0;
}