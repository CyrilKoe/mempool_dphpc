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

// Each core computes the output matrix. No parallelization yet for benchmark reference.
void max_pooling_sequential(int32_t const *__restrict__ A,
                        int32_t const *__restrict__ B, int32_t *__restrict__ C,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t S) {

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
                    if (A[x + k_x + (y + k_y) * M] > max) {
                        max = A[x + k_x + (y + k_y) * M];
                    }
                }
            }
            B[int(x/S) + int(y/S) * (M - int(K/S) + 1)] = max;
        }
    }
  
}