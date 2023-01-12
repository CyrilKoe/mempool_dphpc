#datas_to_bench="4 64 200"
#cores_to_bench="2 4 8 16 32 64 128"

datas_to_bench="4 64 200"
cores_to_bench="2 4 8 16 32 64 128"

progs_to_bench="argmax_impl"

for iteration in {1..20}
do
    for PROG_TO_BENCH in $progs_to_bench
    do
        for DATA_GEN_SIZE in $datas_to_bench
        do
            SEED=$(date +%s)
            MAX=$(echo ${SEED}%10000 | bc)
            (cd argmax_utils && make DATA_GEN_SIZE=${DATA_GEN_SIZE} DATA_GEN_SEED=${SEED} MAX=${MAX} data.h -B)

            # Bench the base case
            RESULT_PATH="results/argmax_base_1_${DATA_GEN_SIZE}_${MAX}_${SEED}"

            make argmax_base
            #(cd ../../hardware/ && app=onnx/argmax_base make simcvcs && make resultpath=${RESULT_PATH} trace)

            for NUM_CORES_BENCH in $cores_to_bench
            do
                RESULT_PATH="results/${PROG_TO_BENCH}_${NUM_CORES_BENCH}_${DATA_GEN_SIZE}_${MAX}_${SEED}"
                make NUM_CORES_BENCH=${NUM_CORES_BENCH} ${PROG_TO_BENCH}
                (cd ../../hardware/ && app=onnx/${PROG_TO_BENCH} make simcvcs && make resultpath=${RESULT_PATH} trace)
            done
        done
    done
done
