from os import walk
import re

result_path = "../../hardware/results"

f = []
for (dirpath, dirnames, filenames) in walk(result_path):
    for filename in filenames:
        if filename != "transcript":
            continue
        with open(dirpath+"/"+filename, "r") as fp:
            for line in fp.readlines():
                if "Benchmark" not in line:
                    continue
                p = re.compile("[0-9]+")
                num_cores, num_data, num_local_data = [a.group() for a in p.finditer(line)]



