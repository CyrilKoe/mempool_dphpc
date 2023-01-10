#!/usr/bin/env python3

# Copyright 2022 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

# This script takes a set of .csv files in one of the results folders and
# generates the average performances over all the cores used.
# Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>

import os
import pandas as pd
import numpy as np

ext = ('.csv')
path = 'reduceSum'
os.chdir(path)
main_path = os.getcwd()
# print(main_path)

df = pd.DataFrame()
cycles = np.zeros(1)
print(cycles)

for folders in (os.listdir(main_path)):
    # print(folders)
    os.chdir(folders)
    path = os.getcwd()
    # print(path)
    for files in os.listdir(path):
        if files.endswith(ext):
            csvread = pd.read_csv(files)

            for section in set(csvread['section']):
                # print("Section %d:\n" % section)
                sectionread = csvread.loc[csvread['section'] == section]
                key = 'cycles'
                column = sectionread[key].replace(np.nan, 0)
                column = column.to_numpy()
                max = np.max(column)
                # print("%-30s %4.4f" % (key, max))
                cycles[section] = max
            df_tmp = pd.DataFrame(cycles, index=['cycles'], columns=[folders])
            df = pd.concat([df, df_tmp], axis=1)
    os.chdir(main_path)
print(df)
df.to_csv('../reduceSum.csv')
