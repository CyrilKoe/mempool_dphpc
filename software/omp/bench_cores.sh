datas_to_bench="4 8 16 32 128"
cores_to_bench="256"
progs_to_bench="my_argmax_dyn_3"

for iteration in {1..4}
do
    for PROG_TO_BENCH in $progs_to_bench
    do
        for DATA_GEN_SIZE in $datas_to_bench
        do
            SEED=$(date +%s)
            MAX=$(echo ${SEED}%2500 | bc)
            (cd my_argmax_data && make DATA_GEN_SIZE=${DATA_GEN_SIZE} DATA_GEN_SEED=${SEED} MAX=${MAX} data.h -B)

            # Bench the base case
            RESULT_PATH="results/my_argmax_dyn_base_1_${DATA_GEN_SIZE}_${MAX}_${SEED}"

            make my_argmax_dyn_base
            (cd ../../hardware/ && app=omp/my_argmax_dyn_base make simcvcs && make resultpath=${RESULT_PATH} trace)

            for NUM_CORES_BENCH in $cores_to_bench
            do
                RESULT_PATH="results/${PROG_TO_BENCH}_${NUM_CORES_BENCH}_${DATA_GEN_SIZE}_${MAX}_${SEED}"
                make NUM_CORES_BENCH=${NUM_CORES_BENCH} ${PROG_TO_BENCH}
                (cd ../../hardware/ && app=omp/${PROG_TO_BENCH} make simcvcs && make resultpath=${RESULT_PATH} trace)
            done
        done
    done
done
