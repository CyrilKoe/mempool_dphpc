import re
import json
import sys
import os
import pandas as pd

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

def reduce_spaces(string):
    if len(string) == 0:
        return string
    buf = '0'
    result = ""
    for char in string:
        if buf == ' ' and char == ' ':
            continue
        result += char
        buf = char
    if result[0] == ' ':
        result = result[1:]
    if result[-1] == ' ':
        result = result[:-1]
    return result

traces_dir = "/scratch/cykoenig/mempool/hardware/results/20221118_111601_my_argmax_dyn_3_f58a5e1/"
trace_files = [traces_dir+f for f in os.listdir(traces_dir) if os.path.isfile(os.path.join(traces_dir, f))]

all_fences_accesses = []
fence_acceses_per_addr = {}

def parse_traces(tmp_file):
    for trace_file in trace_files:
        if not 'trace_hart_' in trace_file:
            continue
        print(trace_file)
        with open(trace_file, 'r') as fp:
            for line in fp.readlines():
                line = line.replace('\n', '')
                line = reduce_spaces(line)
                words = line.split(" ")
                if len(words) > 2 and words[2] == "0x80000334": # amoadd
                    addr = re.search('Word\[(.*)]', words[16], re.IGNORECASE).group(1)
                    cycle = words[0]
                    core_id = trace_file[-8:-6]
                    if addr not in fence_acceses_per_addr:
                        fence_acceses_per_addr[addr] = []
                    all_fences_accesses.append([cycle, addr, core_id])
    with open(tmp_file, "w") as fp:
        fp.write(json.dumps(all_fences_accesses))

def parse_tmp_file(tmp_file):
    with open(tmp_file, "r") as fd:
        str_content = fd.read()
    for a in json.loads(str_content):
        all_fences_accesses.append([int(a[0]), a[1], a[2]])

tmp_file = "traces_parsed.tmp"
if not os.path.exists(tmp_file):
    parse_traces(tmp_file=tmp_file)
else:
    parse_tmp_file(tmp_file=tmp_file)


for a in sorted(all_fences_accesses, key = lambda a : a[0] ):
    print(a)