// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <string.h>

#include "alloc.h"
#include "encoding.h"
#include "libgomp.h"
#include "printf.h"
#include "reduceSum.h"
#include "runtime.h"
#include "synchronization.h"

// Enable verbose printing
#define VERBOSE

// ----------------------------------------------------------------------------
// Input parameters
// ----------------------------------------------------------------------------
#define IN_SIZE 131072
#define OUT_SIZE 256

// Above rank 4, only reduction over all axes is supported
uint32_t shape[] = {256, 512};
uint32_t axes[] = {1};
uint32_t rank = 2;
uint32_t num_axes = 1; // size of axes array
uint32_t noop_with_empty_axes = 0;
// keepdims defaults to 1

int32_t data[IN_SIZE] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));
int32_t reduced[OUT_SIZE] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));

// Initialize the matrices in parallel
void init_vector(int32_t *vector, uint32_t num_elements, uint32_t core_id,
                 uint32_t num_cores) {
  // Parallelize over rows
  for (uint32_t i = core_id; i < num_elements; i += num_cores) {
    vector[i] = 1;
  }
}

void print_vector(int32_t const *vector, uint32_t num_elements) {
  printf("0x%8X\n", (uint32_t)vector);
  for (uint32_t i = 0; i < num_elements; ++i) {
    printf("%5d ", vector[i]);
    printf("\n");
  }
}

int32_t reduce_Sum_wrapper(int32_t const *__restrict__ data,
                           uint32_t *__restrict__ shape,
                           uint32_t *__restrict__ axes, uint32_t rank,
                           uint32_t num_axes, uint32_t keepdims,
                           uint32_t noop_with_empty_axes) {

  int32_t error = 0;
  if (num_axes == 0) {
    if (noop_with_empty_axes) {
      for (int i = 0; i < OUT_SIZE; i++) {
        reduced[i] = data[i];
      }
    } else { // Reduce all axes
      error = reduce_Sum_all(data, shape, rank, reduced);
    }
  } else {
    if (rank == 1) {
      error = reduce_Sum_all(data, shape, rank, reduced);
    } else if (rank == 2) {
      uint32_t inter_shape[4] = {shape[0], shape[1], 1, 1};
      if (num_axes == 1) {
        // printf("Start 2d_1ax...\n");
        error = reduce_Sum_4d_1ax(data, inter_shape, axes[0], reduced);
      } else if (num_axes == 2) {
        // printf("Start 2d_2ax...\n");
        error = reduce_Sum_all(data, shape, rank, reduced);
      } else {
        printf("Num axes cannot exceed rank!\n");
        return 1;
      }
    } else if (rank == 3) {
      uint32_t inter_shape[4] = {shape[0], shape[1], shape[2], 1};
      if (num_axes == 1) {
        // printf("Start 3d_1ax...\n");
        error = reduce_Sum_4d_1ax(data, inter_shape, axes[0], reduced);
      } else if (num_axes == 2) {
        // printf("Start 3d_2ax...\n");
        error = reduce_Sum_4d_2ax(data, inter_shape, axes, reduced);
      } else if (num_axes == 3) {
        // printf("Start 3d_3ax...\n");
        error = reduce_Sum_all(data, shape, rank, reduced);
      } else {
        printf("Num axes cannot exceed rank!\n");
        return 1;
      }
    } else if (rank == 4) {
      if (num_axes == 1) {
        // printf("Start 4d_1ax...\n");
        error = reduce_Sum_4d_1ax(data, shape, axes[0], reduced);
      } else if (num_axes == 2) {
        // printf("Start 4d_2ax...\n");
        error = reduce_Sum_4d_2ax(data, shape, axes, reduced);
      } else if (num_axes == 3) {
        // printf("Start 4d_3ax...\n");
        error = reduce_Sum_4d_3ax(data, shape, axes, reduced);
      } else if (num_axes == 4) {
        // printf("Start 4d_4ax...\n");
        error = reduce_Sum_all(data, shape, rank, reduced);
      } else {
        printf("Num axes cannot exceed rank!\n");
        return 1;
      }
    } else {
      printf("Rank not supported");
      return 1;
    }
  }
  return error;
}

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_timer_t cycles;

  // Initialize synchronization variables
  mempool_barrier_init(core_id);

  // Initialization
  // mempool_init(core_id, num_cores);

#ifdef VERBOSE
  if (core_id == 0) {
    printf("Initialize\n");
  }
#endif

  // Initialize Matrices
  init_vector(data, IN_SIZE, core_id, num_cores);

  // #ifdef VERBOSE
  //   mempool_barrier(num_cores);
  //   if (core_id == 0) {
  //     print_vector(data, M);
  //   }
  // #endif

  mempool_barrier(num_cores);
  int32_t error = 0;

  if (core_id == 0) {
    mempool_wait(4 * num_cores);
    cycles = mempool_get_timer();
    mempool_start_benchmark();
    error = reduce_Sum_wrapper(data, shape, axes, rank, num_axes, 1,
                               noop_with_empty_axes);
    mempool_stop_benchmark();
    cycles = mempool_get_timer() - cycles;

#ifdef VERBOSE
    printf("Error: %d\n", error);
    printf("Duration: %d\n", cycles);
    // print_vector(reduced, OUT_SIZE);
#endif
  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }

  return 0;
}
