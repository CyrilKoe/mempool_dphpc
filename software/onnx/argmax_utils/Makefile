DATA_GEN_SIZE ?= 64
DATA_GEN_SEED ?= 11581
MAX ?= 100
SIZE = $(DATA_GEN_SIZE)*256

data.h:
	../../../scripts/my_gen_data.py --clangformat=clang-format -o data.h --variable=l2_data --size=$(SIZE) --seed=$(DATA_GEN_SEED) --min=-$(MAX) --max=$(MAX)