import re
import json
import sys
import os

trace_path = "/scratch/cykoenig/mempool/hardware/build/traces/trace_hart_0x000000a0"

# ----------------- Architecture Info  -----------------
MEM_REGIONS = {'Other': 0, 'Sequential': 1, 'Interleaved': 2}
NUM_CORES = int(os.environ.get('num_cores', 256))
NUM_TILES = NUM_CORES / 4
SEQ_MEM_SIZE = 4 * int(os.environ.get('seq_mem_size', 1024))
TCDM_SIZE = 16 * 1024 * NUM_TILES

def addr_to_meta(address):
    region = MEM_REGIONS['Other']
    tile = -1
    print(address, SEQ_MEM_SIZE, NUM_TILES)
    if (address < SEQ_MEM_SIZE * NUM_TILES):
        # Local memory
        region = MEM_REGIONS['Sequential']
        tile = address // SEQ_MEM_SIZE
    elif (address < TCDM_SIZE):
        # Interleaved memory
        region = MEM_REGIONS['Interleaved']
        tile = address // 64
        tile = tile % NUM_TILES
    return region, tile

with open(trace_path, "r") as trace_file:
    for line in trace_file:
        _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
        line = _RE_COMBINE_WHITESPACE.sub(" ", line).strip()

        words = line.split(" ")
        if words[2] == "0x80000290":
            str_dict = '{' + line.split('{')[1][:-3].replace('\'', '\"').replace("\": ", "\": \"").replace(", ", "\", ") + '"}'
            words_dict = json.loads(str_dict)
            print(words_dict['opa'])
            print(addr_to_meta(int(words_dict['opa'], base=16)))
