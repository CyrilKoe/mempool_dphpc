// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Viviane Potocnik, ETH Zurich

#include "xpulp/builtins_v2.h"

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
 * Note that all the matrices dimensions must be multiples of 4; these
 * kernels do not have clean-up code and remaining elements would not be
 * considered, leading to wrong results
 */

/*
 * 2D Maxpooling ----------------------------------
 * kernel     = pooling_unrolled_parallel_i8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = yes
 * simd       = yes
 */

dump(maxp, 7);

void pooling_unrolled_parallel_i8_rv32im(int8_t const *__restrict__ A,
                                         int8_t *const __restrict__ B,
                                         uint32_t Mx, uint32_t My, uint32_t K,
                                         uint32_t S, uint32_t core_id,
                                         uint32_t num_cores) {

  const uint32_t x_split = 4;
  uint32_t core_id_x = core_id / x_split;
  uint32_t core_id_y = core_id % x_split;
  uint32_t x_start = core_id_x * S;
  uint32_t y_start = core_id_y * S;
  uint32_t x_end = Mx - K + 1;
  uint32_t y_end = My - K + 1;

  v4s max_vec0;
  v4s max_vec1;
  v4s max_vec2;
  v4s max_vec3;

  v4s a0;
  v4s a1;
  v4s a2;
  v4s a3;

  int8_t max;

  for (uint32_t y = y_start; y < y_end; y += S * x_split) {
    for (uint32_t x = x_start; x < x_end; x += S * num_cores / x_split) {
      // Initialize the maximum with the minimum representable value
      max_vec0 = (v4s)((int32_t *)A)[((x + 0 * 4 + 0) + (y + 0) * Mx) / 4];

      max_vec1 = (v4s)((int32_t *)A)[((x + 1 * 4 + 0) + (y + 0) * Mx) / 4];

      max_vec2 = (v4s)((int32_t *)A)[((x + 2 * 4 + 0) + (y + 0) * Mx) / 4];

      max_vec3 = (v4s)((int32_t *)A)[((x + 3 * 4 + 0) + (y + 0) * Mx) / 4];

      // Iterate over the pooling kernel to find the maximum
      // value inside a pool
      for (uint32_t k_y = 1; k_y < K; k_y++) {
        for (uint32_t k_x = 0; k_x < K / 4; k_x += 4) {
          // if (A[y + k_y + (x + k_x) * M] > max) {
          //     max = A[y + k_y + (x + k_x) * M];
          // }

          a0 = (v4s)((int32_t *)A)[((x + 0 * 4 + k_x) + (y + k_y) * Mx) / 4];

          a1 = (v4s)((int32_t *)A)[((x + 1 * 4 + k_x) + (y + k_y) * Mx) / 4];

          a2 = (v4s)((int32_t *)A)[((x + 2 * 4 + k_x) + (y + k_y) * Mx) / 4];

          a3 = (v4s)((int32_t *)A)[((x + 3 * 4 + k_x) + (y + k_y) * Mx) / 4];

          max_vec0 = __builtin_pulp_max4(a0, max_vec0);
          max_vec1 = __builtin_pulp_max4(a1, max_vec1);
          max_vec2 = __builtin_pulp_max4(a2, max_vec2);
          max_vec3 = __builtin_pulp_max4(a3, max_vec3);
          // dump_maxp((uint32_t)max_vec0);
          // dump_maxp((uint32_t)max_vec1);
          // dump_maxp((uint32_t)max_vec2);
          // dump_maxp((uint32_t)max_vec3);
        }
      }

      // reduce from four to two
      max_vec0 = __builtin_pulp_max4(max_vec0, max_vec1);
      max_vec2 = __builtin_pulp_max4(max_vec2, max_vec3);

      // reduce to single v4s
      max_vec1 = __builtin_pulp_max4(max_vec2, max_vec0);

      max = max_vec1[0];

      // reduce to single int8_t
      for (uint8_t i = 1; i < 4; i++) {
        if (max < max_vec1[i]) {
          max = max_vec1[i];
        }
      }

      // dump_maxp(max);
      // write the maximum to the output matrix
      B[(int)(x / S) + (int)(y / S) * ((int)((Mx - K) / S) + 1)] = max;
    }
  }
}