#!/usr/bin/env bash

# Copyright 2021 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

export mempool_path=../
export app=reduceSum
export script_path=$mempool_path/onnx_benchmark
export folder_path=$mempool_path/onnx_benchmark/reduceSum

mkdir $folder_path

for i in 4 8 16 32 64 128 256 512
do
  export input_size=$i

  cd $mempool_path/software/onnx
  make $app IS=$input_size  
  save_path=$folder_path/$input_size

  cd $mempool_path/hardware
  make simcvcs app=onnx/$app
  export result_dir=$save_path
  make trace
done
