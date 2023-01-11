// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Victor Jung, ETH Zurich

#include "xpulp/builtins_v2.h"
#include <stdint.h>
#include <string.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define BYTES_PER_BANK 64
#define NUMBER_OF_BANKS 8
#define INTERLEAVED_OFFSET 128

// Global locks
uint32_t printLock __attribute__((section(".l1")));

dump(id,1);

// Take a lock
void do_lock(uint32_t *lock) {
  uint32_t islocked;
  islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  while (islocked) {
    asm volatile("nop" ::);
    asm volatile("nop" ::);
    asm volatile("nop" ::);
    asm volatile("nop" ::);
    islocked = __atomic_fetch_or(lock, 1, __ATOMIC_SEQ_CST);
  }
}

// Release a lock
void do_unlock(uint32_t *lock) {
  __atomic_fetch_and(lock, 0, __ATOMIC_SEQ_CST);
}

/*
 * 1D Convolution v1 ----------------------------------
 * kernel               = conv1d_i8
 * data type            = 8-bit integer
 * multi-core           = yes
 * interleaved          = yes
 * kernel duplication   = yes 
 * unrolling            = no
 * simd                 = no
 */
void conv1d_i8_v1(
  int8_t volatile * In_Img, // Input layout: (Batch x Input_Channels x Length)
  int32_t volatile * Out_Img, // Output layout: (Batch x Output_Channels x Length)
  int8_t const * Kernel, // Kernel layout: (Output_Channels x Input_Channels x Length)
  uint32_t const inputLength,
  uint32_t const kernelLength,
  uint32_t const pad_begin, // padding at the beginning of the spatial axis
  uint32_t const pad_end,  // padding at the end of the spatial axis
  uint32_t const strides,
  uint32_t const id,
  uint32_t const numCores
  ) {

  uint32_t const outputLength = (pad_begin + inputLength + pad_end - kernelLength) / strides;
  uint32_t const chunk_size = outputLength / numCores;
  uint32_t output_counter = 0;

  // do_lock(&printLock);
  //printf("In (kernel) %u %u:\n", id*chunk_size*strides, (id+1)*strides*chunk_size);
  // printf("\n");
  // do_unlock(&printLock);

  // Victor: WARNING: We assume that the input is already padded
  output_counter = id*chunk_size;
  for(int i = id*chunk_size*strides; i < (id+1)*strides*chunk_size; i += strides){
    for(int j = 0; j < kernelLength; j++){
      Out_Img[output_counter] += In_Img[i + j] * Kernel[j];
    }
    output_counter++;
  }
}

/*
 * 1D Convolution v2 ----------------------------------
 * kernel               = conv1d_i8
 * data type            = 8-bit integer
 * multi-core           = yes
 * interleaved          = yes
 * kernel duplication   = yes 
 * unrolling            = no
 * simd                 = yes
 */
void conv1d_i8_v2(
  int8_t volatile *__restrict__ In_Img, // Input layout: (Batch x Input_Channels x Length)
  int32_t volatile *__restrict__ Out_Img, // Output layout: (Batch x Output_Channels x Length)
  int8_t const *__restrict__ Kernel, // Kernel layout: (Output_Channels x Input_Channels x Length)
  uint32_t const inputLength,
  uint32_t const kernelLength,
  uint32_t const pad_begin, // padding at the beginning of the spatial axis
  uint32_t const pad_end,  // padding at the end of the spatial axis
  uint32_t const strides,
  uint32_t const id,
  uint32_t const numCores
  ) {

  uint32_t const outputLength = (pad_begin + inputLength + pad_end - kernelLength) / strides;
  uint32_t const chunk_size = outputLength / numCores;
  uint32_t output_counter = 0;

  // printf("In (kernel):\n");

  int32_t * in = (int32_t *)In_Img;
  int32_t * ker = (int32_t *)Kernel;

  // Victor: WARNING: We assume that the input is already padded
  output_counter = id*chunk_size;
  for(int i = id*chunk_size*strides; i < (id+1)*strides*chunk_size; i += strides){
    for(int j = 0; j < kernelLength/4; j++){
      //Out_Img[output_counter] += In_Img[i + j] * Kernel[j];
      Out_Img[output_counter] = __builtin_pulp_sdotsp4((v4s)*(in + (i/4) + j), (v4s)*(ker + j), Out_Img[output_counter]);
    }
    output_counter++;
  }
}


/*
 * 1D Convolution v3 ----------------------------------
 * kernel               = conv1d_i8
 * data type            = 8-bit integer
 * multi-core           = yes
 * interleaved          = yes
 * kernel duplication   = yes
 * unrolling            = yes
 * simd                 = yes
 */
void conv1d_i8_v3(
  int8_t volatile *__restrict__ In_Img, // Input layout: (Batch x Input_Channels x Length)
  int32_t volatile *__restrict__ Out_Img, // Output layout: (Batch x Output_Channels x Length)
  int8_t const *__restrict__ Kernel, // Kernel layout: (Output_Channels x Input_Channels x Length)
  uint32_t const inputLength,
  uint32_t const kernelLength,
  uint32_t const pad_begin, // padding at the beginning of the spatial axis
  uint32_t const pad_end,  // padding at the end of the spatial axis
  uint32_t const strides,
  uint32_t const id,
  uint32_t const numCores
  ) {

  uint32_t const outputLength = (pad_begin + inputLength + pad_end - kernelLength) / strides;
  uint32_t const chunk_size = outputLength / numCores;
  uint32_t output_counter = 0;

  int32_t * in = (int32_t *)In_Img;
  int32_t * ker = (int32_t *)Kernel;

  // Victor: WARNING: We assume that the input is already padded
  output_counter = id*chunk_size;
  for(int i = id*chunk_size*strides; i < (id+1)*strides*chunk_size; i += strides){
    Out_Img[output_counter] = __builtin_pulp_sdotsp4((v4s)*(in + (i/4)), (v4s)*(ker), Out_Img[output_counter]);
    Out_Img[output_counter] = __builtin_pulp_sdotsp4((v4s)*(in + (i/4) + 1), (v4s)*(ker + 1), Out_Img[output_counter]);
    Out_Img[output_counter] = __builtin_pulp_sdotsp4((v4s)*(in + (i/4) + 2), (v4s)*(ker + 2), Out_Img[output_counter]);
    Out_Img[output_counter] = __builtin_pulp_sdotsp4((v4s)*(in + (i/4) + 3), (v4s)*(ker + 3), Out_Img[output_counter]);
    output_counter++;
  }
}


void initializeKernelL1(int8_t* kernel){

  kernel[0] = 42;
  kernel[1] = 16;
  kernel[2] = -95;
  kernel[3] = 11;

  kernel[4] = 106;
  kernel[5] = 75;
  kernel[6] = -12;
  kernel[7] = -69;

  kernel[8] = -34;
  kernel[9] = 1;
  kernel[10] = 15;
  kernel[11] = 0;

  kernel[12] = -91;
  kernel[13] = 29;
  kernel[14] = -43;
  kernel[15] = 54;
}

void initializeInputL1(int8_t* volatile in, uint32_t input_length, uint32_t padBegin, uint32_t padEnd){

  for(int i = 0; i < padBegin + input_length + padEnd; i++) {
    in[i] = 0;
  }
  for(int i = padBegin; i < input_length - padEnd; i++) {
    in[i] = i % 42;
  }
}

void initializeOutputL1(int32_t* volatile out, uint32_t output_length){

  for(int i = 0; i < output_length; i++) {
    out[i] = 0;
  }
}