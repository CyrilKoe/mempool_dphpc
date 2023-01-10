// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Victor Jung, ETH Zurich

#include <stdint.h>
#include <string.h>

#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "kernel/conv1d_i8.h"

// Default Input Parameters for Strong Scaling study
#define INPUT_LENGTH 65536
#define INPUT_CHANNELS 1
#define KERNEL_LENGTH 16

#define OUTPUT_LENGTH 16384
#define OUTPUT_CHANNELS 1

#define PAD_BEGIN 8
#define PAD_END 8
#define STRIDES 4

#define DATA_SCALING_BENCHMARK
// #define STRONG_SCALING_BENCHMARK

volatile int8_t in[PAD_BEGIN + INPUT_LENGTH + PAD_END] __attribute__((section(".l1_prio"))); // interleaved
volatile int32_t out[OUTPUT_LENGTH] __attribute__((section(".l1_prio")));
// int8_t kernel[KERNEL_LENGTH] __attribute__((section(".l1_prio")));
volatile int error __attribute__((section(".l1")));

int main() {
  uint32_t core_id = mempool_get_core_id();
  uint32_t num_cores = mempool_get_core_count();

  int8_t kernel[KERNEL_LENGTH];

  mempool_barrier_init(core_id);
  mempool_barrier(num_cores);


  //////////////////////////// Initialisation ////////////////////////////
  if (core_id == 0) {
    // Initialize error
    error = 0;
    // Initialize Input
    printf("Init Input\n");
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + INPUT_LENGTH; i < PAD_BEGIN + INPUT_LENGTH + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + INPUT_LENGTH; i++) {
      in[i] = i % 42;
    }
    // Initialize output
    printf("Init Output\n");
    // initializeOutputL1(out, OUTPUT_LENGTH);
    for(int i = 0; i < OUTPUT_LENGTH; i++) {
      out[i] = 0;
    }
    // Initialize kernel
    printf("Init Kernel\n");
    // initializeKernelL1(kernel);
  }
  initializeKernelL1(kernel);


  //////////////////////////// Debug Print ////////////////////////////
  if (core_id == 0){ 
    printf("In:\n");
    for (int j = 255; j < 255 + 15; j++) {
      printf("%d ", in[j]);
    }
    printf("\n");
    printf("Ker:\n");
    for (int j = 0; j < 10; j++) {
      printf("%d ", kernel[j]);
    }
    printf("\n");
  }

  #ifdef STRONG_SCALING_BENCHMARK
  //////////////////////////// Strong Scaling Benchmark ////////////////////////////
  int32_t volatile numCoreBenchmark[6] = {1, 8, 16, 64, 128, 256};
  mempool_barrier(num_cores);


  if (core_id < numCoreBenchmark[0]) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, INPUT_LENGTH, KERNEL_LENGTH,
                  PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreBenchmark[0]);
    mempool_stop_benchmark();
    }
  mempool_barrier(num_cores);
  if(core_id == 0){
    printf("Output kernel:\n");
    for (int j = 10; j < 20; j++) {
      printf("%d ", out[j]);
    }
    printf("\n");
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u core benchmark\n", numCoreBenchmark[1]);
    for(int i = 0; i < OUTPUT_LENGTH; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreBenchmark[1]) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, INPUT_LENGTH, KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreBenchmark[1]);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u core benchmark\n", numCoreBenchmark[2]);
    for(int i = 0; i < OUTPUT_LENGTH; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreBenchmark[2]) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, INPUT_LENGTH, KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreBenchmark[2]);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u core benchmark\n", numCoreBenchmark[3]);
    for(int i = 0; i < OUTPUT_LENGTH; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreBenchmark[3]) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, INPUT_LENGTH, KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreBenchmark[3]);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u core benchmark\n", numCoreBenchmark[4]);
    for(int i = 0; i < OUTPUT_LENGTH; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreBenchmark[4]) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, INPUT_LENGTH, KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreBenchmark[4]);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u core benchmark\n", numCoreBenchmark[5]);
    for(int i = 0; i < OUTPUT_LENGTH; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreBenchmark[5]) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, INPUT_LENGTH, KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreBenchmark[5]);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);
  #endif


  #ifdef DATA_SCALING_BENCHMARK

  //////////////////////////// Data Scaling Benchmark ////////////////////////////
  int32_t volatile inputLengthBenchmark[7] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
  int32_t volatile outputLengthBenchmark[7] = {256, 512, 1024, 2048, 4096, 8192, 16384};
  uint32_t volatile numCoreDataBenchmark = 256; // 1
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[0]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[0]; i < PAD_BEGIN + inputLengthBenchmark[0] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[0]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[0]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[0], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[1]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[1]; i < PAD_BEGIN + inputLengthBenchmark[1] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[1]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[1]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[1], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[2]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[2]; i < PAD_BEGIN + inputLengthBenchmark[2] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[2]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[2]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[2], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);
  

  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[3]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[3]; i < PAD_BEGIN + inputLengthBenchmark[3] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[3]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[3]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[3], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[4]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[4]; i < PAD_BEGIN + inputLengthBenchmark[4] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[4]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[4]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[4], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[5]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[5]; i < PAD_BEGIN + inputLengthBenchmark[5] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[5]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[5]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[5], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);


  if(core_id == 0){
    printf("Start v1 kernel %u input size benchmark\n", inputLengthBenchmark[6]);
    // Initialize Input
    for(int i = 0; i < PAD_BEGIN; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN + inputLengthBenchmark[6]; i < PAD_BEGIN + inputLengthBenchmark[6] + PAD_END; i++) {
      in[i] = 0;
    }
    for(int i = PAD_BEGIN; i < PAD_BEGIN + inputLengthBenchmark[6]; i++) {
      in[i] = (int8_t)i % 42;
    }
    for(int i = 0; i < outputLengthBenchmark[6]; i++) {
      out[i] = 0;
    }    
  }
  mempool_barrier(num_cores);
  if (core_id < numCoreDataBenchmark) {
    mempool_start_benchmark();
    conv1d_i8_v3(in, out, kernel, (uint32_t)inputLengthBenchmark[6], KERNEL_LENGTH,
                 PAD_BEGIN, PAD_END, STRIDES, core_id, numCoreDataBenchmark);
    mempool_stop_benchmark();
  }
  mempool_barrier(num_cores);
  #endif


  mempool_barrier(num_cores);

  return error;
}
