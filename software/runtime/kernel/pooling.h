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
#define K_END 16
#define STRIDE 4

// for now hard-coded, but should be OUT_SIZE + K - S
#define M_DIM 268
#define X_START 0
#define X_END 253//(M_DIM - K + 1)
#define Y_START 0
#define Y_END 253//(M_DIM - K + 1)

dump(max, 7);
dump(checksum, 8);
dump(index, 9);
// Each core computes the output matrix. No parallelization yet for benchmark reference.
void static inline max_pooling_sequential(int32_t const *__restrict__ A, int32_t *const __restrict__ B,
                        uint32_t M, uint32_t K, uint32_t S) {

    // Initialize the maximum with the minimum representable value
    int32_t volatile max0;
    int32_t volatile max1;
    int32_t volatile max2;
    int32_t volatile max3;
    // int32_t checksum = 0;
    
    for (uint32_t x = 0; x < M - K + 1; x += S) {
        for (uint32_t y = 0; y < M - K + 1; y += S) {
            // Initialize the maximum with the minimum representable value
            max0 = -2147483648;
            max1 = -2147483648;
            max2 = -2147483648;
            max3 = -2147483648;
            // Iterate over the pooling kernel to find the maximum
            // value inside a pool
            for (uint32_t k_x = 0; k_x < K; k_x++) {
                for (uint32_t k_y = 0; k_y < K / 4; k_y += 4) {
                    // if (A[y + k_y + (x + k_x) * M] > max) {
                    //     max0 = A[y + k_y + 0 + (x + k_x) * M];
                    //     max1 = A[y + k_y + 1 + (x + k_x) * M];
                    //     max2 = A[y + k_y + 2 + (x + k_x) * M];
                    //     max3 = A[y + k_y + 3 + (x + k_x) * M];
                    // }

                    max0 = A[y + k_y + 0 + (x + k_x) * M] > max0 ? A[y + k_y + 0 + (x + k_x) * M] : max0;
                    max1 = A[y + k_y + 1 + (x + k_x) * M] > max1 ? A[y + k_y + 1 + (x + k_x) * M] : max1;
                    max2 = A[y + k_y + 2 + (x + k_x) * M] > max2 ? A[y + k_y + 2 + (x + k_x) * M] : max2;
                    max3 = A[y + k_y + 3 + (x + k_x) * M] > max3 ? A[y + k_y + 3 + (x + k_x) * M] : max3;
                }
            }
            // Matrix below for writing back, but will not be benchmarked atm
            // FIXME: indices not correct yet
            // write back to output matrix

            for (int i = 0; i < 4; i++) {
                if (max0 > max1) {
                    max1 = max0;
                }
                if (max2 > max3) {
                    max3 = max2;
                }
                if (max1 > max3) {
                    max3 = max1;
                }
            }

            B[(int)(x/S) + (int)(y/S) * ((int)((M - K)/S) + 1)] = max3;

            // printf("Maximum value = %d\n", max);
            // dump_max(max);
            // checksum += max;
        }

        // dump_index(x);
    }

    // dump_checksum(checksum);
}

// parallelizing computation over all cores
void max_pooling_parallel(int32_t const *__restrict__ A, int32_t *const __restrict__ B,
                        uint32_t Mx, uint32_t My, uint32_t K, uint32_t S, 
                        uint32_t core_id, uint32_t num_cores) {

    int32_t max_0;
    int32_t max_1;
    int32_t max_2;
    int32_t max_3;
    
    const uint32_t x_split = 4;
    uint32_t core_id_x = core_id / x_split;
    uint32_t core_id_y = core_id % x_split;
    uint32_t x_start = core_id_x * S;
    uint32_t y_start = core_id_y * S;
    uint32_t x_end = Mx - K + 1; 
    uint32_t y_end = My - K + 1;
    
    for (uint32_t y = y_start; y < y_end; y += S * x_split) {
        for (uint32_t x = x_start; x < x_end; x += S * num_cores / x_split) {
            // Initialize the maximum with the minimum representable value
            max_0 = -2147483648;
            max_1 = -2147483648;
            max_2 = -2147483648;
            max_3 = -2147483648;
            // Iterate over the pooling kernel to find the maximum
            // value inside a pool
            for (uint32_t k_y = 0; k_y < K; k_y++) {
                for (uint32_t k_x = 0; k_x < K ; k_x += 4) {
                    // if (A[x + k_x + (y + k_y) * Mx] > max) { // change to x + k_x + (y + k_y) * Mx
                    //     max = A[x + k_x + (y + k_y) * Mx]; // change to x + k_x + (y + k_y) * Mx
                    // }

                    max_0 = A[x + k_x + 0 + (y + k_y) * Mx] > max_0 ? A[x + k_x + 0 + (y + k_y) * Mx] : max_0;
                    max_1 = A[x + k_x + 1 + (y + k_y) * Mx] > max_1 ? A[x + k_x + 1 + (y + k_y) * Mx] : max_1;
                    max_2 = A[x + k_x + 2 + (y + k_y) * Mx] > max_2 ? A[x + k_x + 2 + (y + k_y) * Mx] : max_2;
                    max_3 = A[x + k_x + 3 + (y + k_y) * Mx] > max_3 ? A[x + k_x + 3 + (y + k_y) * Mx] : max_3;
                }
            }

            max_0 = __builtin_pulp_maxsi(max_0, max_1);
            max_2 = __builtin_pulp_maxsi(max_2, max_3);
            max_0 = __builtin_pulp_maxsi(max_0, max_2);
            B[(int)(x/S) + (int)(y/S) * ((int)((Mx - K)/S) + 1)] = max_0; // change to x/S + y/S * (Mx - K + 1)/S
            // dump_max(max);
        }
    }
  
}


void static inline max_pooling_openmp_static(int32_t const *__restrict__ A, int32_t *const __restrict__ B,
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
            B[(int)(x/S) + (int)(y/S) * ((int)((M - K)/S) + 1)] = max;
            // dump_max(max);
        }
    }
  
}

void static inline max_pooling_openmp_dynamic(int32_t const *__restrict__ A, int32_t *const __restrict__ B,
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
            B[(int)(x/S) + (int)(y/S) * ((int)((M - K)/S) + 1)] = max;
            // dump_max(max);
        }
    }
  
}