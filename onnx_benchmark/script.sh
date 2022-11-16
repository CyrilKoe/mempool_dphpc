#!/usr/bin/env bash

export mempool_path=/scratch2/gislamoglu/hpc/mempool
export app=onnx_ReduceSum
export script_path=$mempool_path/onnx_benchmark
export folder_path=$mempool_path/onnx_benchmark/ReduceSum_new

mkdir $folder_path

for i in 4 8 16 32 64 128 256 512
do
  export input_size=$i

  cd $mempool_path/software/omp
  make $app IS=$input_size  
  save_path=$folder_path/$input_size

  cd $mempool_path/hardware
  make simcvcs app=omp/$app
  export result_dir=$save_path
  make trace
done
