DATA_GEN_SIZE ?= 64
SIZE = $(DATA_GEN_SIZE)*256
MAX = 100

data.h:
	../../../scripts/my_gen_data.py --clangformat=clang-format -o data.h --variable=l2_data --size=$(SIZE) --min=-$(MAX) --max=$(MAX)