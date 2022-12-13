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

/*
 * 1D Convolution v1 ----------------------------------
 * kernel               = conv1d_i8
 * data type            = 8-bit integer
 * multi-core           = yes
 * interleaved          = yes
 * kernel duplication   = no
 * unrolling            = no
 * simd                 = no
 */
void conv1d_i8_v1(
  int8_t volatile *__restrict__ In_Img, // Input layout: (Batch x Input_Channels x Length)
  int32_t volatile *__restrict__ Out_Img, // Output layout: (Batch x Output_Channels x Length)
  int8_t const *__restrict__ Kernel, // Kernel layout: (Output_Channels x Input_Channels x Length)
  uint32_t const batchSize,
  uint32_t const inputLength,
  uint32_t const numberOfInputChanels,
  uint32_t const kernelLength,
  uint32_t const corePerKernel,
  uint32_t const numberOfOutputChannels,
  uint32_t const pad_begin, // padding at the beginning of the spatial axis
  uint32_t const pad_end,  // padding at the end of the spatial axis
  uint32_t const strides,
  uint32_t const id,
  uint32_t const numCores
  ) {

  uint32_t const outputLength = pad_begin + inputLength + pad_end - kernelLength;
  uint16_t const chunk_size = outputLength*numberOfInputChanels / numCores;
  //uint32_t const kernelLocationOffset = (id/corePerKernel)*kernelLength;

  uint32_t output_counter = 0;

  // Victor: WARNING: We assume that the input is already padded
  for(int k = 0; k < numberOfOutputChannels; k++){
    output_counter = k*kernelLength + id*chunk_size;
    for(int i = id*chunk_size*strides; i < (id+1)*strides*chunk_size; i += strides){
      for(int j = k*kernelLength*numberOfInputChanels; j < (k+1)*kernelLength*numberOfInputChanels; j++){
        Out_Img[output_counter] = In_Img[i + j] * Kernel[j];
      }
      output_counter++;
    }
  }
}

/*
 * 1D Convolution v2 ----------------------------------
 * kernel     = conv1d_i8
 * data type  = 8-bit integer
 * multi-core = yes
 * interleaved = yes
 * unrolling 1  = no
 * unrolling 2  = no
 * simd       = yes
 */
void conv1d_i8_v2(
  int8_t volatile *__restrict__ In_Img, // Input layout: (Batch x Input_Channels x Length)
  int32_t volatile *__restrict__ Out_Img, // Output layout: (Batch x Output_Channels x Length)
  int8_t const *__restrict__ Kernel, // Kernel layout: (Output_Channels x Input_Channels x Length)
  uint32_t const batchSize,
  uint32_t const inputLength,
  uint32_t const numberOfInputChanels,
  uint32_t const kernelLength,
  uint32_t const corePerKernel,
  uint32_t const numberOfOutputChannels,
  uint32_t const pad_begin, // padding at the beginning of the spatial axis
  uint32_t const pad_end,  // padding at the end of the spatial axis
  uint32_t const strides,
  uint32_t const id,
  uint32_t const numCores
  ) {

  uint32_t const outputLength = (pad_begin + inputLength + pad_end - kernelLength) / strides;
  uint16_t const chunk_size = outputLength*numberOfInputChanels / numCores;
  uint32_t const kernelLocationOffset = (id/corePerKernel)*kernelLength;

  uint32_t output_counter = 0;

  v4s Im_vector1;
  v4s Im_vector2;
  v4s Im_vector3;
  v4s Im_vector4;
  v4s kernel_vector1 = (v4s){Kernel[0 + kernelLocationOffset], Kernel[1 + kernelLocationOffset], Kernel[2 + kernelLocationOffset], Kernel[3 + kernelLocationOffset]};
  v4s kernel_vector2 = (v4s){Kernel[4 + kernelLocationOffset], Kernel[5 + kernelLocationOffset], Kernel[6] + kernelLocationOffset, Kernel[7 + kernelLocationOffset]};
  v4s kernel_vector3 = (v4s){Kernel[8 + kernelLocationOffset], Kernel[9 + kernelLocationOffset], Kernel[10 + kernelLocationOffset], Kernel[11 + kernelLocationOffset]};
  v4s kernel_vector4 = (v4s){Kernel[12 + kernelLocationOffset], Kernel[13 + kernelLocationOffset], Kernel[14 + kernelLocationOffset], Kernel[15 + kernelLocationOffset]};

  // WARNING: We assume that the input is already padded
  for(int k = 0; k < numberOfOutputChannels; k++){
    output_counter = k*outputLength + id*chunk_size;
    for(int i = id*chunk_size*strides; i < (id+1)*strides*chunk_size; i += strides) {
      Im_vector1 = (v4s){In_Img[i + 0], In_Img[i + 1], In_Img[i + 2], In_Img[i + 3]};
      Im_vector2 = (v4s){In_Img[i + 4], In_Img[i + 5], In_Img[i + 6], In_Img[i + 7]};
      Im_vector3 = (v4s){In_Img[i + 8], In_Img[i + 9], In_Img[i + 10], In_Img[i + 11]};
      Im_vector4 = (v4s){In_Img[i + 12], In_Img[i + 13], In_Img[i + 14], In_Img[i + 15]};

      Out_Img[output_counter] = __builtin_pulp_dotsp4(Im_vector1, kernel_vector1);
      Out_Img[output_counter] += __builtin_pulp_dotsp4(Im_vector2, kernel_vector2);
      Out_Img[output_counter] += __builtin_pulp_dotsp4(Im_vector3, kernel_vector3);
      Out_Img[output_counter] += __builtin_pulp_dotsp4(Im_vector4, kernel_vector4);
      output_counter++;
    }
  }
}