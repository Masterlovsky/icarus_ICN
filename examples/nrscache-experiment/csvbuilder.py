#! /usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# A Script to dump the parameter and results from result_pickle file
# Created on 2023-5-28, powered by Masterlovsky
# Version: 1.0
# """
from collections import defaultdict
from icarus.results.readwrite import *
import pandas as pd
import sys


def main(input_f, output_f):
    """
    Dump the results from result files, save to csv
    :output_f: the output file name
    :input_f: the input file name
    :return: None
    """
    pickle_data = read_results_pickle(input_f)
    df = pd.DataFrame(columns=["methods", "alpha", "workload", "CHR", "free_space"])
    for i, experiment in enumerate(pickle_data):
        # get useful data and save to pandas dataframe
        desc = experiment[0]["desc"]
        params = desc.split("/")
        method = params[1].split(":")[1].strip()
        alpha = eval(params[2].split(":")[1].strip())
        workload = eval(params[3].split(":")[1].strip())
        cache_hit_ratio = experiment[1]["CACHE_HIT_RATIO"]["MEAN"]
        free_space = experiment[1]["FREE_SPACE"]["AVG_FREE_SPACE_RATIO"]
        df.loc[i] = [method, alpha, workload, cache_hit_ratio, free_space]
    # group by method, sort by alpha, then sort by workload
    df = df.groupby(["methods"]).apply(lambda x: x.sort_values(["alpha", "workload"]))
    # remove index
    df = df.reset_index(drop=True)
    # add column "mean_chr" which is the mean of cache hit ratio in workloads
    df["mean_chr"] = df.groupby(["methods", "alpha"])["CHR"].transform("mean")
    # add column "mean_free_space" which is the mean of free space in workloads
    df["mean_free_space"] = df.groupby(["methods", "alpha"])["free_space"].transform("mean")
    # add column "std_chr" which is the std of cache hit ratio in workloads
    df["std_chr"] = df.groupby(["methods", "alpha"])["CHR"].transform("std")
    # add column "std_free_space" which is the std of free space in workloads
    df["std_free_space"] = df.groupby(["methods", "alpha"])["free_space"].transform("std")
    df.to_csv(output_f, index=False)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 csvbuilder.py <result_pickle_file> <output_file>")
        exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
    print("Build csv done!")
