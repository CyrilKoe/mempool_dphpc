import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from mako import template
import tqdm

TEMPLATE_FILE = './templates/data.mako'
SEEDS_DIR = './seeds'
OUTPUT_DIR = './headers'

def best_case(n : int, precision : int, signed : bool):
    assert precision == 8 or precision == 16 or precision == 32, 'unsupported data precision'
    min_val = 0
    max_val = 2 ** precision - 1
    if signed:
        shift = 2 ** (precision - 1)
        min_val -= shift
        max_val -= shift
    assert n < (max_val - min_val), "n too large"
    return [min_val + i for i in range(0, n)]

def worst_case(n : int, precision : int, signed : bool):
    assert precision == 8 or precision == 16 or precision == 32, 'unsupported data precision'
    min_val = 0
    max_val = 2 ** precision - 1
    if signed:
        shift = 2 ** (precision - 1)
        min_val -= shift
        max_val -= shift
    assert n < (max_val - min_val), "n too large"
    return [max_val - i for i in range(0, n)]

def generate_data(n : int, precision : int, signed : bool, seed : int):
    '''
    Generates random array of <n> elements with <precision> bits integer values using <seed>
    as random seed.
    '''
    random.seed(seed)
    assert precision == 8 or precision == 16 or precision == 32, 'unsupported data precision'
    min_val = 0
    max_val = 2 ** precision - 1
    if signed:
        shift = 2 ** (precision - 1)
        min_val -= shift
        max_val -= shift
    values = [random.randint(min_val, max_val) for _ in range(n)]
    return values

def dump_to_hex(values : List[int], k : int, precision : int, signed : bool, filename : str):
    '''
    Generates C header file from Mako template.
    '''
    temp = template.Template(filename=TEMPLATE_FILE)
    with open(filename, 'w') as f:
        f.write(temp.render(values=values, k=k, precision=precision, signed=signed))

def plot_insertions(insertions : np.ndarray, filename : str = './insertions.png'):
    print('MEAN: %.2f' % np.mean(insertions))
    print('STD : %.2f' % np.std(insertions))
    print('MIN : %d' % np.min(insertions))
    print('MAX : %d' % np.max(insertions))
    plt.hist(insertions, bins=int(np.max(insertions)-np.min(insertions)))
    plt.savefig(filename)

def measure_insertions(iterations, n, k, precision):
    '''
    Counts the average number of insertions performed by the Top-K algorithm for the given
    hyperparameters (N, K) over <iterations> iterations. Each iteration is run using a different
    random seed, to extract representative statistics.
    '''
    values_range = 2 ** precision
    insertions = np.zeros((iterations, ), dtype=int)
    seeds = []
    for i in tqdm.tqdm(range(iterations)):
        #seed = random.randint(0, 100000)
        seed = i
        seeds.append(seed)
        random.seed(seed)
        topk = [random.randint(0, values_range) for _ in range(k)]
        topk.sort()
        count = 0
        for _ in range(k, n):
            val = random.randint(0, values_range)
            if val > topk[0]:
                topk[0] = val
                topk.sort()
                count += 1
        insertions[i] = count
    return insertions

def generate_seeds_filename(n, k):
    '''
    Creates a unique name for the .npy seeds file for each (n, k) pair.
    '''
    return 'seeds_n%d_k%d' % (n, k)

def generate_seeds(ns, ks, precision, nseeds, iterations):
    '''
    Computes and saves <nseeds> random seeds in the <SEEDS_DIR> directory for 
    each pair of values in ns, ks. The <nseeds> seeds are selected such that the 
    resulting input vectors of size n generated with those seeds, lead to an average
    number of insertions in the top-k algorithm (with respect to all possible seeds
    in the [0, <iterations>) range)
    '''
    
    def gen_values_around_mean(mean, radius):
        r1 = range(mean + 1, mean + radius,  1)
        r2 = range(mean - 1, mean - radius, -1)
        r = [mean] + [el for pair in zip(r1, r2) for el in pair]
        r_noneg = [val for val in filter(lambda x : x >= 0, r)]
        return r_noneg
    
    for n in ns:
        for k in ks:
            print('Measuring insertions for N=%d, K=%d' % (n, k))
            insertions = measure_insertions(iterations, n, k, precision)
            mean = np.mean(insertions)
            seeds = []
            targets = gen_values_around_mean(int(mean), radius=nseeds)
            for target in targets:
                idxs, = np.where(insertions == target)
                for idx in idxs:
                    seeds.append(idx)
                if len(seeds) >= nseeds:
                    break
            seeds = seeds[:nseeds]
            filename = generate_seeds_filename(n, k)
            np.save(os.path.join(SEEDS_DIR, filename), np.array(seeds).astype(int))

def generate_header(seed : int, n, k, precision, signed, filepath):
    '''
    Creates a C header file named <header_file> that contains a randomly generated input
    vector of size <n> using the random seed <seed>.
    '''
    values = generate_data(n, precision, signed, int(seed))
    dump_to_hex(values, k, precision, signed, filepath)

def generate_headers(ns, ks, precision, signed, directory):
    '''
    Generate C header files for all pairs of Ns and Ks.

    This is a helper function that generates sample input vectors, combining all values
    in ns, ks and all the pre-computed seeds in SEEDS_DIR for each (n, k) pair.  
    '''
    for n in ns:
        for k in ks:
            filename = generate_seeds_filename(n, k) + '.npy'
            seeds = np.load(os.path.join(SEEDS_DIR, filename)).astype(int)
            print('Generating headers for testcase (N=%d, K=%d)' % (n, k))
            for seed in seeds:
                filename = 'n%d_k%d_seed%d.h' % (n, k, seed)
                filepath = os.path.join(directory, filename)
                generate_header(int(seed), n, k, precision, signed, filepath)

##################################################
## PROFILING NUMBER OF INSERTIONS PERFORMED     ##
## BY TOPK DEPENDING ON INPUT DATA DISTRIBUTION ##
##################################################
# assuming the topk array starts prefilled with
# the first K elements of the input array
## -> number of insertions is in range [0, N-K]
## -> best case  : 0 updates
## -> worst case : N-K updates
## -> expectation: unknown, measure statistically

# ITERATIONS = 10
# NSEEDS = 10
# PRECISION = 32
# SIGNED = False

def make_dirs(outdir):
    if not os.path.exists(SEEDS_DIR):
        os.mkdir(SEEDS_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.exists(outdir):
        os.mkdir(outdir)


################################# DATA GENERATION SCRIPT ####################################
# Generates sample input data for the Top-K algorithm.                                      #
# The data generation is split into 2 phases:                                               #
#                                                                                           #
# 1. Representative values computation:                                                     #
#    During this phase, for each hyper-parameter configuration (N, K, precision, signed),   #
#    the expected number insertions performed by the top-k is measured for a set of         #
#    randomly generated values. The first NSEEDS values that generate such representative   #
#    data are chosen and stored in a file (named with the generate_seeds_filename function) #
#    in the SEEDS_DIR directory.                                                            #
#                                                                                           #
# 2. Data generation                                                                        #
#    The previously generated seeds are read from the filesystem to generate representative #
#    input data for the Top-K algorithm. The data is saved in the form of a set of C header #
#    files, that can then be included in the C source code. The header files are created in #
#    a directory named according to the data type, within the OUTDIR directory.             #
#############################################################################################


def main(action, iterations, nseeds, precision, signed):
    # name output directory according to data type (precision + sign)
    outdir = os.path.join(OUTPUT_DIR, '%sint%d' % ('' if signed else 'u', precision))
    make_dirs(outdir) # create all directories
    def ki(n):
        return n * 1024
    
    # 0. Define the Ns and Ks ranges that will be used to generate data
    ns = range(ki(8), ki(128), ki(8))
    ks = range(8, 64+1, 8)
    
    # 1. Compute and store the representative random seeds for all Ns and Ks
    #    WARNING: This function can take a long time to execute for large number
    #             of iterations. This can be reduced at the cost of lower precision.
    if action != 'genheaders':
        print('Generating seeds...')
        # generate_seeds(ns, ks, precision, nseeds, iterations)

    # 2. Generate the header files in the <outdir> directory
    if action != 'genseeds':
        print('Generating headers...')
        # generate_headers(ns, ks, precision, signed, outdir)

    print('Done!\nThe generated header files can be found in %s' % outdir)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate representative random data for Top-K')
    parser.add_argument('action', type=str, choices={'genseeds', 'genheaders', 'all'})
    parser.add_argument('--iterations', '-i', type=int , default=10000, required=False)
    parser.add_argument('--precision' , '-p', type=int , default=   32, required=False)
    parser.add_argument('--signed'    , '-s', type=bool, default=False, required=False)
    parser.add_argument('--nseeds'    , '-n', type=int , default=   10, required=False)
    args = parser.parse_args()
    main(args.action, args.iterations, args.nseeds, args.precision, args.signed)
