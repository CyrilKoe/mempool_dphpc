from os import walk
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

result_path = "../../hardware/results"

configs = []
tot_df = pd.DataFrame()
for (dirpath, dirnames, filenames) in walk(result_path):
    num_cores = -1
    num_data = -1
    num_local_data = -1
    max_val = -1
    seed = -1
    for filename in filenames:
        if filename != "transcript":
            continue
        p = re.compile("[0-9]+")
        # Get parameters from the directory name first 
        num_cores, num_data, max_val, seed = [a.group() for a in p.finditer(dirpath.split('/')[4])][-4:]
        num_cores, num_data, max_val, seed = int(num_cores), int(num_data), int(max_val), int(seed)
        nul_local_data = int(num_data / num_cores)
        # Override them based on transcript if any
        with open(dirpath+"/"+filename, "r") as fp:
            for line in fp.readlines():
                if "Benchmark" not in line:
                    continue
                p = re.compile("[0-9]+")
                num_cores, num_data, num_local_data = [a.group() for a in p.finditer(line)]
                num_cores, num_data, num_local_data = int(num_cores), int(num_data), int(num_local_data)
                if "ERROR" in line:
                    print("ERROR")
                    assert(0)

        # Read CSV
        df = pd.read_csv(dirpath+"/results.csv", index_col=0)
        df["num_cores"] = int(num_cores)
        df["num_data"] = int(num_data)
        df["num_local_data"] = int(num_local_data)
        df["max_val"] = int(max_val)
        df["seed"] = int(seed)
        # Concat everyone
        configs.append([num_cores, num_data, num_local_data])
        tot_df = pd.concat([tot_df, df], axis=0, join="outer", ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)

# Drop nans
to_drop = tot_df.columns[tot_df.isnull().any()]
for nan_col in to_drop:
    assert(not nan_col in ['cycles', 'num_data', 'num_cores'])
    tot_df = tot_df.drop(nan_col, axis=1)

# Drop strings
tot_df = tot_df.select_dtypes(exclude=['object'])

tot_df.to_csv("ok.csv")


print(tot_df)

colors = [(2, "green"), (8, "orange"), (64, "blue"), (200, "brown")]

fig, ax = plt.subplots(constrained_layout=True)

for num_data, color in colors: #tot_df['num_data'].unique():
    experiment_df = tot_df.loc[tot_df['num_data'] == num_data]
    Y1, Y1_std, Y2, Y2_std = [], [], [], []
    
    num_cores_arr = [2, 4, 8, 16, 32, 64, 128, 256]
    for num_cores in num_cores_arr:
        res_argmax = []
        res_reduction = []
        for seed in experiment_df['seed'].unique():
            mask = (experiment_df['seed'] == seed) & (experiment_df['num_cores'] == num_cores)
            masked = experiment_df[mask]
            if masked.empty:
                continue
            res_argmax.append(    masked.loc[masked['section'] == 0]['cycles'].max() / 1000000 )
            res_reduction.append( masked.loc[masked['section'] == 1]['cycles'].max() / 1000000 )
        Y1.append(         np.mean(np.array(res_argmax))   )
        Y1_std.append(  10*np.std(np.array(res_argmax))    )
        Y2.append(         np.mean(np.array(res_reduction)))
        Y2_std.append(  10*np.std(np.array(res_reduction)) )
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    plt.figure()
    plt.errorbar(num_cores_arr, Y1, yerr=Y1_std, fmt='o')
    plt.errorbar(num_cores_arr, Y1+Y2, yerr=Y2_std, fmt='o')
    plt.ylim([0, np.max(Y1+Y2+Y2_std+Y2_std)])
    plt.title(str(num_data * 256) + " datas (int32)")
    #point = 2*(num_data*256) / (1024 - 1)
    #if int(point) != 1:
    #    plt.axvline(x = point,linestyle="--",color=color)

plt.show()
assert(0)
plt.title("Execution time for "+str(colors[0][0]*256)+" int32")
plt.yscale('linear')
plt.ylabel('seconds')
plt.xlabel('# cores')
new_tick_locations = tot_df['num_cores'].unique()
plt.xticks(new_tick_locations)


plt.savefig("figure_1.png")