// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Gamze Islamoglu, ETH Zurich

#include "encoding.h"
#include "printf.h"
#include <stddef.h>
#include <stdint.h>

int32_t reduce_SumSquare_omp_static(int32_t const *__restrict__ A,
                              uint32_t num_elements) {
  uint32_t i;
  int32_t reduced = 0;
#pragma omp parallel for reduction(+ : reduced)
  for (i = 0; i < num_elements; i++) {
    reduced += A[i]*A[i];
  }
  return reduced;
}

int32_t reduce_SumSquare_all(int32_t const *__restrict__ data,
                          uint32_t *__restrict__ shape, 
                          uint32_t rank,
                          int32_t *__restrict__ reduced) {
  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < rank; i++){
    num_elements *= shape[i];
  }
  *reduced = reduce_SumSquare_omp_static(data, num_elements);
  return 0;
}

int32_t reduce_SumSquare_4d_1ax(int32_t const *__restrict__ data,
                          uint32_t *__restrict__ shape, 
                          uint32_t ax,
                          int32_t *__restrict__ reduced) {
  uint32_t I0, I1;
  uint32_t B0, B1, B2, B3;
  uint32_t R0, R1, R2, R3;
  uint32_t I = shape[0];
  uint32_t J = shape[1];
  uint32_t K = shape[2];
  uint32_t L = shape[3];

  if (ax == 0){
    I0 = K*L;
    I1 = L;
    B0 = K*L;
    B1 = L;
    B2 = 1;
    B3 = J*K*L;
    R0 = J;
    R1 = K;
    R2 = L;
    R3 = I;
  }
  else if (ax == 1){
    I0 = K*L;
    I1 = L;
    B0 = J*K*L;
    B1 = L;
    B2 = 1;
    B3 = K*L;
    R0 = I;
    R1 = K;
    R2 = L;
    R3 = J;
  }
  else if (ax == 2){
    I0 = J*L;
    I1 = L;
    B0 = J*K*L;
    B1 = K*L;
    B2 = 1;
    B3 = L;
    R0 = I;
    R1 = J;
    R2 = L;
    R3 = K;
  }
  else if (ax == 3){
    I0 = J*K;
    I1 = K;
    B0 = J*K*L;
    B1 = K*L;
    B2 = L;
    B3 = 1;
    R0 = I;
    R1 = J;
    R2 = K;
    R3 = L;
  }
  else{
    printf("Invalid axis! \n");
    return 1;
  }

#pragma omp parallel for collapse(3) num_threads(NTHREADS)
  for (uint32_t i0 = 0; i0 < R0; i0++){
    for (uint32_t i1 = 0; i1 < R1; i1++){
      for (uint32_t i2 = 0; i2 < R2; i2++){
        reduced[i0*I0 + i1*I1 + i2] = 0;
        // printf("core_id = %d\n", mempool_get_core_id());
        for (uint32_t i3 = 0; i3 < R3; i3++){
          reduced[i0*I0 + i1*I1 + i2] += data[i0*B0 + i1*B1 + i2*B2 + i3*B3]*data[i0*B0 + i1*B1 + i2*B2 + i3*B3];
        }
      }
    }
  }
  return 0;
}

int32_t reduce_SumSquare_4d_2ax(int32_t const *__restrict__ data,
                          uint32_t *__restrict__ shape, 
                          uint32_t *__restrict__ ax,
                          int32_t *__restrict__ reduced) {

  int32_t error = 0;
  uint32_t inter_size = 1;
  uint32_t inter_shape[4];

  for (uint32_t i = 0; i < 4; i++){
    if (i != ax[0]){
      inter_size *= shape[i];
      inter_shape[i] = shape[i];
    } else {
      inter_shape[i] = 1;
    }
  }

  int32_t inter_reduced[inter_size];

  error = reduce_SumSquare_4d_1ax(data, shape, ax[0], inter_reduced);
  error = reduce_SumSquare_4d_1ax(inter_reduced, inter_shape, ax[1], reduced);
  return error;
}

int32_t reduce_SumSquare_4d_3ax(int32_t const *__restrict__ data,
                          uint32_t *__restrict__ shape, 
                          uint32_t *__restrict__ ax,
                          int32_t *__restrict__ reduced) {

  int32_t error = 0;                          
  uint32_t inter1_size = 1;
  uint32_t inter2_size = 1;
  uint32_t inter_shape[4];

  for (uint32_t i = 0; i < 4; i++){
    if (i != ax[0]){
      inter1_size *= shape[i];
      inter_shape[i] = shape[i];
      if (i != ax[1]){
        inter2_size *= shape[i];
      }
    } else {
      inter_shape[i] = 1;
    }
  }

  int32_t inter1_reduced[inter1_size];
  int32_t inter2_reduced[inter2_size];

  error = reduce_SumSquare_4d_1ax(data, shape, ax[0], inter1_reduced);
  error = reduce_SumSquare_4d_1ax(inter1_reduced, inter_shape, ax[1], inter2_reduced);
  inter_shape[ax[1]] = 1;
  error = reduce_SumSquare_4d_1ax(inter2_reduced, inter_shape, ax[2], reduced);
  return error;
}
