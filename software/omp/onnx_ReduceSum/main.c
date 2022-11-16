// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <string.h>

#include "encoding.h"
#include "kernel/mat_mul.h"
#include "libgomp.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

// Define Vector dimensions:
// C = AB with A=[Mx1], B=[Mx1]
#define M (IS * NUM_CORES) // IS_max = 512
// Specify how the vectors A and B should be initialized
// The entries will follow this format:
// a(i) = A_a*i + A_b
// b(i) = B_a*i + B_b
// The result will be the following
// c = (A_b*B_b) * N
//     + (A_a*B_b+A_b*B_a) * (N*(N-1))/2
//     + (A_a*B_a) * (N*(N-1)*(2*N-1))/6
// Note: To keep the code simpler, we use indices that go from 0 to N-1 instead
// of 1 to N as the mathematicians do. Hence, for A, i=[0,M-1].
#define A_a 1
#define A_b 1
// Enable verbose printing
#define VERBOSE

int32_t volatile init __attribute__((section(".l2"))) = 0;
int32_t a[M] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));
int32_t c[NUM_CORES] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));
int32_t reduced_group[4] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));
int32_t partial[4 * NUM_CORES] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));
int32_t reduced16[16] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));
int32_t reduced4[4] __attribute__((section(".l1")))
__attribute__((aligned(NUM_CORES * 4 * 4)));

// Initialize the matrices in parallel
void init_vector(int32_t *vector, uint32_t num_elements, int32_t a, int32_t b,
                 uint32_t core_id, uint32_t num_cores) {
  // Parallelize over rows
  for (uint32_t i = core_id; i < num_elements; i += num_cores) {
    vector[i] = a * (int32_t)i + b;
  }
}

void print_vector(int32_t const *vector, uint32_t num_elements) {
  printf("0x%8X\n", (uint32_t)vector);
  for (uint32_t i = 0; i < num_elements; ++i) {
    printf("%5d ", vector[i]);
    printf("\n");
  }
}

int32_t reduce_sum_sequential(int32_t const *__restrict__ A,
                              uint32_t num_elements) {
  uint32_t i;
  int32_t reduced = 0;
  for (i = 0; i < num_elements; i++) {
    reduced += A[i];
  }
  return reduced;
}

int32_t reduce_sum_parallel1(int32_t const *__restrict__ A,
                              int32_t *__restrict__ Partial_sums,
                              uint32_t num_elements, uint32_t id,
                              uint32_t numThreads) {

  Partial_sums[id] = 0;
  int32_t reduced = 0;
  for (uint32_t i = id; i < num_elements/4; i += numThreads) {
    for (uint32_t j = 0; j < 4; ++j) {
      Partial_sums[id] += A[i*4+j];
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    for (uint32_t i = 0; i < numThreads; i += 1) {
      reduced += Partial_sums[i];
    }
  }
  mempool_barrier(numThreads);
  return reduced;
}

// does not make sense to me since memory layout is not considered
int32_t reduce_sum_parallel2(int32_t const *__restrict__ A,
                              int32_t *__restrict__ Partial_sums,
                              uint32_t num_elements, uint32_t id,
                              uint32_t numThreads) {

  Partial_sums[id] = 0;
  int32_t reduced = 0;
  for (uint32_t i = id * num_elements / numThreads;
       i < (id + 1) * num_elements / numThreads; i += 1) {
    Partial_sums[id] += A[i];
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    for (uint32_t i = 0; i < numThreads; i += 1) {
      reduced += Partial_sums[i];
    }
  }
  mempool_barrier(numThreads);
  return reduced;
}

int32_t reduce_sum_parallel3(int32_t const *__restrict__ A,
                              int32_t *__restrict__ Partial_sums,
                              uint32_t num_elements, uint32_t id,
                              uint32_t numThreads) {

  Partial_sums[id] = 0;
  int32_t reduced = 0;
  for (uint32_t i = id; i < num_elements/4; i += numThreads) {
    for (uint32_t j = 0; j < 4; ++j) {
      Partial_sums[id] += A[i*4+j];
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    reduced_group[0] = 0;
    for (uint32_t i = id; i < 64; i += 1) {
      reduced_group[0] += Partial_sums[i];
    }
  }
  if (id == 64) {
    reduced_group[1] = 0;
    for (uint32_t i = id; i < 128; i += 1) {
      reduced_group[1] += Partial_sums[i];
    }
  }
  if (id == 128) {
    reduced_group[2] = 0;
    for (uint32_t i = id; i < 192; i += 1) {
      reduced_group[2] += Partial_sums[i];
    }
  }
  if (id == 192) {
    reduced_group[3] = 0;
    for (uint32_t i = id; i < 256; i += 1) {
      reduced_group[3] += Partial_sums[i];
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    for (uint32_t i = 0; i < 4; i += 1) {
      reduced += reduced_group[i];
    }
  }
  mempool_barrier(numThreads);
  return reduced;
}

int32_t reduce_sum_parallel4(int32_t const *__restrict__ A,
                              int32_t *__restrict__ Partial_sums,
                              uint32_t num_elements, uint32_t id,
                              uint32_t numThreads) {

  Partial_sums[id] = 0;
  int32_t reduced = 0;
  for (uint32_t i = id; i < num_elements/4; i += numThreads) {
    for (uint32_t j = 0; j < 4; ++j) {
      Partial_sums[id] += A[i*4+j];
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    reduced_group[0] = 0;
    for (uint32_t i = id; i < 64; i += 1) {
      reduced_group[0] += Partial_sums[i];
    }
  }
  if (id == 16) {
    reduced_group[1] = 0;
    for (uint32_t i = 64; i < 128; i += 1) {
      reduced_group[1] += Partial_sums[i];
    }
  }
  if (id == 32) {
    reduced_group[2] = 0;
    for (uint32_t i = 128; i < 192; i += 1) {
      reduced_group[2] += Partial_sums[i];
    }
  }
  if (id == 48) {
    reduced_group[3] = 0;
    for (uint32_t i = 192; i < 256; i += 1) {
      reduced_group[3] += Partial_sums[i];
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    for (uint32_t i = 0; i < 4; i += 1) {
      reduced += reduced_group[i];
    }
  }
  mempool_barrier(numThreads);
  return reduced;
}

int32_t reduce_sum_parallel5(int32_t const *__restrict__ A,
                              int32_t *__restrict__ Partial_sums,
                              uint32_t num_elements, uint32_t id,
                              uint32_t numThreads) {

  Partial_sums[id] = 0;
  int32_t reduced = 0;
  for (uint32_t i = id; i < num_elements/4; i += numThreads) {
    for (uint32_t j = 0; j < 4; ++j) {
      Partial_sums[id] += A[i*4+j];
    }
  }
  mempool_barrier(numThreads);
  for (uint32_t i = 0; i < 16; i += 1) {
    if (id == i*4) {
      reduced16[i] = 0;
      for (uint32_t j = 4*id; j < 4*id+16; j += 1) {
        reduced16[i] += Partial_sums[j];
      }
    }
  }
  mempool_barrier(numThreads);
  for (uint32_t i = 0; i < 4; i += 1) {
    if (id == i*4) {
      reduced4[i] = 0;
      for (uint32_t j = id; j < id+4; j += 1) {
        reduced4[i] += reduced16[j];
      }
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    for (uint32_t i = 0; i < 4; i += 1) {
      reduced += reduced4[i];
    }
  }
  mempool_barrier(numThreads);
  return reduced;
}

int32_t reduce_sum_omp_static(int32_t const *__restrict__ A,
                              uint32_t num_elements) {
  uint32_t i;
  int32_t reduced = 0;
#pragma omp parallel for reduction(+ : reduced)
  for (i = 0; i < num_elements; i++) {
    reduced += A[i];
  }
  return reduced;
}

int32_t reduce_sum_omp_dynamic(int32_t const *__restrict__ A,
                               uint32_t chunksize) {
  uint32_t i;
  int32_t reduced = 0;
  // printf("num_elements %d\n", num_elements);
#pragma omp parallel for schedule(dynamic, chunksize) reduction(+ : reduced)
  for (i = 0; i < M; i++) {
    reduced += A[i];
  }
  return reduced;
}

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();
  mempool_timer_t cycles;

  // Initialize synchronization variables
  mempool_barrier_init(core_id);

#ifdef VERBOSE
  if (core_id == 0) {
    printf("Initialize\n");
  }
#endif

  // Initialize Matrices
  init_vector(a, M, A_a, A_b, core_id, num_cores);

#ifdef VERBOSE
  mempool_barrier(num_cores);
  if (core_id == 0) {
    // print_vector(a, M);
    // print_vector(b, M);
  }
#endif

  mempool_barrier(num_cores);
  int32_t result, correct_result;

//   if (core_id == 0) {
//     mempool_wait(4 * num_cores);
//     cycles = mempool_get_timer();
//     mempool_start_benchmark();
//     result = reduce_sum_sequential(a, M);
//     mempool_stop_benchmark();
//     cycles = mempool_get_timer() - cycles;
//   }

// #ifdef VERBOSE
//   mempool_barrier(num_cores);
//   if (core_id == 0) {
//     printf("Sequential Result: %d\n", result);
//     printf("Sequential Duration: %d\n", cycles);
//   }
// #endif
//   mempool_barrier(num_cores);

  cycles = mempool_get_timer();
  mempool_start_benchmark();
  result = reduce_sum_parallel1(a, c, M, core_id, num_cores);
  mempool_stop_benchmark();
  cycles = mempool_get_timer() - cycles;

#ifdef VERBOSE
  mempool_barrier(num_cores);
  if (core_id == 0) {
    printf("Manual Parallel1 Result: %d\n", result);
    printf("Manual Parallel1 Duration: %d\n", cycles);
  }
#endif
  mempool_barrier(num_cores);

  cycles = mempool_get_timer();
  mempool_start_benchmark();
  result = reduce_sum_parallel2(a, c, M, core_id, num_cores);
  mempool_stop_benchmark();
  cycles = mempool_get_timer() - cycles;

#ifdef VERBOSE
  mempool_barrier(num_cores);
  if (core_id == 0) {
    printf("Manual Parallel2 Result: %d\n", result);
    printf("Manual Parallel2 Duration: %d\n", cycles);
  }
#endif
  mempool_barrier(num_cores);

  cycles = mempool_get_timer();
  mempool_start_benchmark();
  result = reduce_sum_parallel3(a, c, M, core_id, num_cores);
  mempool_stop_benchmark();
  cycles = mempool_get_timer() - cycles;

#ifdef VERBOSE
  mempool_barrier(num_cores);
  if (core_id == 0) {
    printf("Manual Parallel3 Result: %d\n", result);
    printf("Manual Parallel3 Duration: %d\n", cycles);
  }
#endif
  mempool_barrier(num_cores);

  cycles = mempool_get_timer();
  mempool_start_benchmark();
  result = reduce_sum_parallel4(a, c, M, core_id, num_cores);
  mempool_stop_benchmark();
  cycles = mempool_get_timer() - cycles;

#ifdef VERBOSE
  mempool_barrier(num_cores);
  if (core_id == 0) {
    printf("Manual Parallel4 Result: %d\n", result);
    printf("Manual Parallel4 Duration: %d\n", cycles);
  }
#endif
  mempool_barrier(num_cores);

  /*  OPENMP IMPLEMENTATION  */
  int32_t omp_result;

  if (core_id == 0) {
    mempool_wait(4 * num_cores);

    cycles = mempool_get_timer();
    mempool_start_benchmark();
    omp_result = reduce_sum_omp_static(a, M);
    mempool_stop_benchmark();
    cycles = mempool_get_timer() - cycles;

    printf("OMP Static Result: %d\n", omp_result);
    printf("OMP Static Duration: %d\n", cycles);

    mempool_wait(4 * num_cores);

    cycles = mempool_get_timer();
    mempool_start_benchmark();
    omp_result = reduce_sum_omp_dynamic(a, IS);
    mempool_stop_benchmark();
    cycles = mempool_get_timer() - cycles;

    printf("OMP Dynamic(%d) Result: %d\n", IS, omp_result);
    printf("OMP Dynamic(%d) Duration: %d\n", IS, cycles);

    mempool_wait(4 * num_cores);

    // cycles = mempool_get_timer();
    // mempool_start_benchmark();
    // omp_result = reduce_sum_omp_dynamic(a, 16);
    // mempool_stop_benchmark();
    // cycles = mempool_get_timer() - cycles;

    // printf("OMP Dynamic(16) Result: %d\n", omp_result);
    // printf("OMP Dynamic(16) Duration: %d\n", cycles);

    // mempool_wait(4 * num_cores);

  } else {
    while (1) {
      mempool_wfi();
      run_task(core_id);
    }
  }
  return 0;
}
