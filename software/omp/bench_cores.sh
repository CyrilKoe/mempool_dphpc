datas_to_bench="4 8 16 32 63"
cores_to_bench="32 64 128 256"
progs_to_bench="my_argmax_dyn_3"

for PROG_TO_BENCH in $progs_to_bench
do
    for DATA_GEN_SIZE in $datas_to_bench
    do
        (cd my_argmax_data && make DATA_GEN_SIZE=${DATA_GEN_SIZE} data.h -B)
        for NUM_CORES_BENCH in $cores_to_bench
        do
            #(cd ${PROG_TO_BENCH} && make NUM_CORES_BENCH=${NUM_CORES_BENCH} run -B)
            make NUM_CORES_BENCH=${NUM_CORES_BENCH} ${PROG_TO_BENCH}
            (cd ../../hardware/ && app=omp/${PROG_TO_BENCH} make simcvcs && make trace)
        done
    done
done