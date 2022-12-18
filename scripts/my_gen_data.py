#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import re
import numpy as np

license = """\
// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
"""

script_path = "scripts/gen_data.py"


def gen_header(command):
    header = license
    header += '// Automatically generated by:\n'
    header += f'// {script_path} {command}\n'
    return header


def gen_var(variable, size, min, max, generator):
    dim = [int(x) for x in re.findall(r'\d+', size)]
    num = np.prod(dim)
    val = ','.join(map(str, generator.integers(min, max, size=num)))

    argmax = []
    max_val = int(min) - 1
    for idx, i in enumerate(val.split(',')):
        if int(i) > max_val:
            max_val = int(i)
            argmax = [idx]
        elif int(i) == max_val:
            argmax.append(idx)
    
    var = f'#include <inttypes.h>\n#define DATA_LEN {num}\n'
    var += f'int32_t {variable}_flat[DATA_LEN] __attribute__((section(".l2"))) = {{ {val} }};\n'
    var += f'uint32_t const {variable}_len = DATA_LEN;\n'
    var += f'uint32_t const expected_indexes_len = {len(argmax)};\n'
    var += f'int32_t const expected_global_max = {max_val};\n'
    var += f'//result_max = {max_val}\n// result_len = {len(argmax)} \n// result = {argmax}\n'
    
    offset = 0
    core = 0
    page = int(int(num) / 256)
    while offset != num:
        local_vector = [int(a) for a in val.split(',')[offset:offset+page]]
        var += f'//core : {core} , offset : {offset}  = {str(local_vector)}\n'
        offset += page
        core += 1
    return var


def main():
    # Argument parsing
    parser = argparse.ArgumentParser('gen_data', allow_abbrev=True)
    parser.add_argument(
        '-s',
        '--size',
        nargs='+',
        action='append',
        help='Size of generated variable'
    )
    parser.add_argument(
        '-v',
        '--variable',
        nargs='+',
        action='append',
        help='Variable name')
    parser.add_argument(
        '-o',
        '--output',
        nargs=1,
        help='Output header file')
    parser.add_argument(
        '--min',
        nargs='?',
        default=-2147483648,
        help='Min value')
    parser.add_argument(
        '--max',
        nargs='?',
        default=2147483647,
        help='Max value')
    parser.add_argument(
        '--seed',
        nargs='?',
        default=11581,
        help='Random number seed value')
    parser.add_argument(
        '--clangformat',
        nargs='?',
        default='clang-format',
        help='Clang format executable')
    args = parser.parse_args()

    sizes = args.size
    variables = args.variable
    file = args.output[0]
    clang_format = args.clangformat
    min_val = args.min
    max_val = args.max
    seed = args.seed
    seed = int(seed)

    # print('sizes:', sizes, file=sys.stderr)
    # print('variables:', variables, file=sys.stderr)
    # print('file:', file, file=sys.stderr)
    # print('Command: ', sys.argv, file=sys.stderr)

    generator = np.random.default_rng(seed)

    # Write the file
    with open(file, "w") as f:
        f.write(gen_header(' '.join(sys.argv[1:])))
        for variable, size in zip(variables, sizes):
            var = gen_var(variable[0], size[0], min_val, max_val, generator)
            f.write(var)

    # Format the final file
    os.system(f'{clang_format} -i {file}')


if __name__ == '__main__':
    main()
