from io import StringIO
from os import walk
import re
import sys
import pandas as pd

args = {'VERBOSE':0, 'REMOTE_PATH':''}
for arg in sys.argv:
    if '=' in arg:
        argname, value = arg.split('=')
        if argname not in args:
            print("Unknown arg",argname)
            sys.exit(1)
        if argname in ['VERBOSE']:
            args[argname] = int(value)
        else:
            args[argname] = value

remote_path = args['REMOTE_PATH']
result_path = remote_path+"../hardware/results/"

def split_line(line):
    cycle_0, cycle_1, addr, instr, opa, opb = [a for a in line.split(' ') if a != ""][0:6]
    return cycle_0, cycle_1, addr, instr, opa, opb

todo = 0
done = 0
for (dirpath, dirnames, filenames) in walk(result_path):
    for filename in filenames:
        if "trace_hart_" in filename:
            todo+=1

print(todo,"files todo")
for (dirpath, dirnames, filenames) in walk(result_path):
    csv_string = "section,"
    to_add = {}
    for filename in filenames:
        if "trace_hart_" in filename:
            if dirpath+'/results.csv' not in to_add:
                cols_to_add = {}
                col_idx=0
                with open(dirpath+'/results.csv', "r") as fp:
                    if "cs_retry" in fp.readline():
                        done+=1
                        if done%100 == 0:
                            print("\r"+str(done)+"/"+str(todo),end="")
                        continue
                to_add[dirpath+'/results.csv'] = cols_to_add
                
            with open(dirpath+"/"+filename, "r") as fp:
                buffer = StringIO()
                sys.stdout = buffer
                
                cs_cycle_0, cs_cycle_1, cs_addr, cs_instr = -1, -1, -1, -1
                retry = 0
                section = 0
                core = int(filename[-10:-6],16)
                
                cols_to_add[col_idx] = {"core": core, "section":section, "cs_retry":[], "cs_duration":[]}
                
                for line in fp.readlines():
                    if "amoor" in line and cs_cycle_0 == -1:
                        cs_cycle_0, cs_cycle_1, cs_addr, cs_instr, _, _ = split_line(line)
                        print("("+str(section)+")","->", cs_addr, cs_instr, end="")
                        
                    elif "amoor" in line and cs_cycle_0 != -1:
                        retry += 1
                        
                    elif "csrwi" in line:
                        cycle_0, cycle_1, addr, instr, csr, value = split_line(line)
                        if csr == "trace,":
                            section = section + 1
                            col_idx+=1
                            cols_to_add[col_idx] = {"core": core, "section":section, "cs_retry":[], "cs_duration":[]}
                            
                    elif "amoswap.w" in line and cs_cycle_0 != -1:
                        cycle_0, cycle_1, addr, instr, _, _ = split_line(line)
                        print("->", addr, instr, "->", int(cycle_1)-int(cs_cycle_1), "("+str(retry)+")")
                        cols_to_add[col_idx]["cs_retry"].append(retry)
                        cols_to_add[col_idx]["cs_duration"].append(int(cycle_1)-int(cs_cycle_1))
                        cs_cycle_0 = -1
                        retry = 0

                if(cs_cycle_0 != -1):
                    # In case an amoor is not followed by an amoswap
                    print("")
                
                print_output = buffer.getvalue()
                sys.stdout = sys.__stdout__
                if print_output and args['VERBOSE']:
                    print("\n",filename)
                    print(print_output)
                done += 1
                if done%100 == 0:
                    print("\r"+str(done)+"/"+str(todo),end="")
 
    for filename in to_add:
        df_add = pd.DataFrame.from_dict(to_add[filename], orient='index')
        df = pd.read_csv(filename)
        # If only core and sections are in common
        if len(df.columns.intersection(df_add.columns)) == 2:
            df = df.merge(df_add, on=['core', 'section'])
        # print("\r",filename)
        df.to_csv(filename, index=False)
print("\r"+str(done)+"/"+str(done),end="")
print("\nDone.")
