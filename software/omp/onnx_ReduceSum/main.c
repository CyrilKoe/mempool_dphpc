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
  if (id == 0) {
    reduced16[0] = 0;
    for (uint32_t j = 0; j < 16; j += 1) {
      reduced16[0] += Partial_sums[j];
    }
  }
  if (id == 4) {
    reduced16[1] = 0;
    for (uint32_t j = 16; j < 32; j += 1) {
      reduced16[1] += Partial_sums[j];
    }
  }
  if (id == 8) {
    reduced16[2] = 0;
    for (uint32_t j = 32; j < 48; j += 1) {
      reduced16[2] += Partial_sums[j];
    }
  }
  if (id == 12) {
    reduced16[3] = 0;
    for (uint32_t j = 48; j < 64; j += 1) {
      reduced16[3] += Partial_sums[j];
    }
  }
  if (id == 16) {
    reduced16[4] = 0;
    for (uint32_t j = 64; j < 80; j += 1) {
      reduced16[4] += Partial_sums[j];
    }
  }
  if (id == 20) {
    reduced16[5] = 0;
    for (uint32_t j = 80; j < 96; j += 1) {
      reduced16[5] += Partial_sums[j];
    }
  }
  if (id == 24) {
    reduced16[6] = 0;
    for (uint32_t j = 96; j < 112; j += 1) {
      reduced16[6] += Partial_sums[j];
    }
  }
  if (id == 28) {
    reduced16[7] = 0;
    for (uint32_t j = 112; j < 128; j += 1) {
      reduced16[7] += Partial_sums[j];
    }
  }
  if (id == 32) {
    reduced16[8] = 0;
    for (uint32_t j = 128; j < 144; j += 1) {
      reduced16[8] += Partial_sums[j];
    }
  }
  if (id == 36) {
    reduced16[9] = 0;
    for (uint32_t j = 144; j < 160; j += 1) {
      reduced16[9] += Partial_sums[j];
    }
  }
  if (id == 40) {
    reduced16[10] = 0;
    for (uint32_t j = 160; j < 176; j += 1) {
      reduced16[10] += Partial_sums[j];
    }
  }
  if (id == 44) {
    reduced16[11] = 0;
    for (uint32_t j = 176; j < 192; j += 1) {
      reduced16[11] += Partial_sums[j];
    }
  }
  if (id == 48) {
    reduced16[12] = 0;
    for (uint32_t j = 192; j < 208; j += 1) {
      reduced16[12] += Partial_sums[j];
    }
  }
  if (id == 52) {
    reduced16[13] = 0;
    for (uint32_t j = 208; j < 224; j += 1) {
      reduced16[13] += Partial_sums[j];
    }
  }
  if (id == 56) {
    reduced16[14] = 0;
    for (uint32_t j = 224; j < 240; j += 1) {
      reduced16[14] += Partial_sums[j];
    }
  }
  if (id == 60) {
    reduced16[15] = 0;
    for (uint32_t j = 240; j < 256; j += 1) {
      reduced16[15] += Partial_sums[j];
    }
  }
  mempool_barrier(numThreads);
  if (id == 0) {
    reduced4[0] = 0;
    for (uint32_t j = 0; j < 4; j += 1) {
      reduced4[0] += reduced16[j];
    }
  }
  if (id == 1) {
    reduced4[1] = 0;
    for (uint32_t j = 4; j < 8; j += 1) {
      reduced4[1] += reduced16[j];
    }
  }
  if (id == 2) {
    reduced4[2] = 0;
    for (uint32_t j = 8; j < 12; j += 1) {
      reduced4[2] += reduced16[j];
    }
  }
  if (id == 3) {
    reduced4[3] = 0;
    for (uint32_t j = 12; j < 16; j += 1) {
      reduced4[3] += reduced16[j];
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

  cycles = mempool_get_timer();
  mempool_start_benchmark();
  result = reduce_sum_parallel5(a, c, M, core_id, num_cores);
  mempool_stop_benchmark();
  cycles = mempool_get_timer() - cycles;

#ifdef VERBOSE
  mempool_barrier(num_cores);
  if (core_id == 0) {
    printf("Manual Parallel5 Result: %d\n", result);
    printf("Manual Parallel5 Duration: %d\n", cycles);
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
