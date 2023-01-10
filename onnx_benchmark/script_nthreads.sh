#!/usr/bin/env bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

export mempool_path=../
export app=reduceSum
export script_path=$mempool_path/onnx_benchmark
export folder_path=$mempool_path/onnx_benchmark/reduceSum

# mkdir $folder_path

for i in 1 2 4 8 16 32 64 128 256
do
  export nthreads=$i

  cd $mempool_path/software/onnx
  make $app NTHREADS=$nthreads  
  save_path=$folder_path/$nthreads

  cd $mempool_path/hardware
  make simcvcs app=onnx/$app
  export result_dir=$save_path
  make trace
done
