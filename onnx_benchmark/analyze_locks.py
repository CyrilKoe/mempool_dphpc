from io import StringIO
from os import walk
import re
import sys

result_path = "/scratch/cykoenig/development/mempool_dphpc/hardware/results/"

def split_line(line):
    cycle_0, cycle_1, addr, instr = [a for a in line.split(' ') if a != ""][0:4]
    return cycle_0, cycle_1, addr, instr

for (dirpath, dirnames, filenames) in walk(result_path):
    for filename in filenames:
        if "trace_hart_" in filename:
            with open(dirpath+"/"+filename, "r") as fp:
                buffer = StringIO()
                sys.stdout = buffer
                
                cs_cycle_0, cs_cycle_1, cs_addr, cs_instr = -1, -1, -1, -1
                retry = 0
                
                for line in fp.readlines():
                    if "amoor" in line and cs_cycle_0 == -1:
                        cs_cycle_0, cs_cycle_1, cs_addr, cs_instr = split_line(line)
                        print("->", cs_addr, cs_instr, end="")
                    elif "amoor" in line and cs_cycle_0 != -1:
                        retry += 1
                    if "amoswap.w" in line and cs_cycle_0 != -1:
                        cycle_0, cycle_1, addr, instr = split_line(line)
                        print(" ->", addr, instr, "->", int(cycle_1)-int(cs_cycle_1), "("+str(retry)+")")
                        cs_cycle_0 = -1
                        retry = 0
                if(cs_cycle_0 != -1):
                    # In case an amoor is not followed by an amoswap
                    print("")
                
                print_output = buffer.getvalue()
                sys.stdout = sys.__stdout__
                if print_output:
                    print(filename)
                    print(print_output)