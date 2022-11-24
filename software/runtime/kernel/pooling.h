// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Viviane Potocnik, ETH Zurich

/* This library implements a naive max pooling in multiple different ways.
 * At the moment we assume quadratic input matrices whose dimensions are
 * integer multiples of the pooling kernel. Further we only use quadratic
 * kernels. The stride has the same size in both C and W direction. 
 * The functions all follow the following format:
 *
 * A is an M x M matrix, B is an N x N matrix. The kernel is a K x K matrix.
 * In detail, the output matrix is of size: N = FLOOR((M - K) / S + 1) where 
 * S determines the stride of the pooling kernels. 
 * 
 */

// Add defines for dynamic scheduling
#define CHUNK 1
#define THREADS 256

// Add defines for broken openMP in mempool
#define K_END 2
#define STRIDE 1
#define OUT 4
// for now hard-coded, but should be OUT_SIZE + K - S
#define M_DIM 5
#define X_START 0
#define X_END 4 //(M_DIM - K + 1)
#define Y_START 0
#define Y_END 4 //(M_DIM - K + 1)

dump(max, 7);
// Each core computes the output matrix. No parallelization yet for benchmark reference.
void max_pooling_sequential(int32_t const *__restrict__ A,
                        uint32_t M, uint32_t K, uint32_t S) {

    // Initialize the maximum with the minimum representable value
    int32_t max;
    
    for (uint32_t x = 0; x < M - K + 1; x += S) {
        for (uint32_t y = 0; y < M - K + 1; y += S) {
            // Initialize the maximum with the minimum representable value
            max = -2147483648;
            // Iterate over the pooling kernel to find the maximum
            // value inside a pool
            for (uint32_t k_x = 0; k_x < K; k_x++) {
                for (uint32_t k_y = 0; k_y < K; k_y++) {
                    if (A[y + k_y + (x + k_x) * M] > max) {
                        max = A[y + k_y + (x + k_x) * M];
                    }
                }
            }
            // Matrix below for writing back, but will not be benchmarked atm
            // FIXME: indices not correct yet
            // B[int(x/S) + int(y/S) * (int((M - K)/S) + 1)] = max;
            printf("Maximum value = %d\n", max);
        }
    }
  
}

// parallelizing computation over all cores
void max_pooling_parallel(int32_t const *__restrict__ A,
                        uint32_t M, uint32_t K, uint32_t S, 
                        uint32_t core_id, uint32_t num_cores) {

    // Initialize the maximum with the minimum representable value
    int32_t max;
    uint32_t x_start = 0;
    uint32_t y_start = core_id * S;
    uint32_t x_end = M - K + 1;
    uint32_t y_end = M - K + 1;
    
    for (uint32_t x = x_start; x < x_end; x += S) {
        for (uint32_t y = y_start; y < y_end; y += S * num_cores) {
            // Initialize the maximum with the minimum representable value
            max = -2147483648;
            // Iterate over the pooling kernel to find the maximum
            // value inside a pool
            for (uint32_t k_x = 0; k_x < K; k_x++) {
                for (uint32_t k_y = 0; k_y < K; k_y++) {
                    if (A[y + k_y + (x + k_x) * M] > max) {
                        max = A[y + k_y + (x + k_x) * M];
                    }
                }
            }
            // B[int(x/S) + int(y/S) * (int((M - K)/S) + 1)] = max;
            dump_max(max);
        }
    }
  
}

void max_pooling_openmp_static(int32_t const *__restrict__ A,
                        uint32_t M, uint32_t K, uint32_t S) {

    // Initialize the maximum with the minimum representable value
    int32_t max;
    #pragma omp parallel for collapse(2) firstprivate(max)
    for (uint32_t x = 0; x < M - K + 1; x += S) {
        for (uint32_t y = 0; y < M - K + 1; y += S) {
            // Initialize the maximum with the minimum representable value
            max = -2147483648;
            // Iterate over the pooling kernel to find the maximum
            // value inside a poolSS
            for (uint32_t k_x = 0; k_x < K; k_x++) {
                for (uint32_t k_y = 0; k_y < K; k_y++) {
                    if (A[y + k_y + (x + k_x) * M] > max) {
                        max = A[y + k_y + (x + k_x) * M];
                    }
                }
            }
            // B[int(x/S) + int(y/S) * (int((M - K)/S) + 1)] = max;
            dump_max(max);
        }
    }
  
}

void max_pooling_openmp_dynamic(int32_t const *__restrict__ A,
                        uint32_t M, uint32_t K, uint32_t S) {

    // Initialize the maximum with the minimum representable value
    int32_t max;
    #pragma omp parallel for schedule(dynamic, CHUNK) num_threads(THREADS) collapse(2) firstprivate(max)
    for (uint32_t x = 0; x < X_END; x += STRIDE) {
        for (uint32_t y = 0; y < Y_END; y += STRIDE) {
            // Initialize the maximum with the minimum representable value
            max = -2147483648;
            // Iterate over the pooling kernel to find the maximum
            // value inside a pool
            for (uint32_t k_x = 0; k_x < K_END; k_x++) {
                for (uint32_t k_y = 0; k_y < K_END; k_y++) {
                    if (A[y + k_y + (x + k_x) * M] > max) {
                        max = A[y + k_y + (x + k_x) * M];
                    }
                }
            }
            // B[int(x/S) + int(y/S) * (int((M - K)/S) + 1)] = max;
            dump_max(max);
        }
    }
  
}