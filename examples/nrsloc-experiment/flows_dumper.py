#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Script to dump the flows data from result files
Created on 2024-1-2, powered by Masterlovsky
Version: 1.0
"""
import sys
from collections import defaultdict

from icarus.results.readwrite import *


def dump_flow_data(input_file: str = "results.pickle", output_file: str = "flow_data.txt"):
    """
    Dump the flows data from result files
    :output_file: the output file name
    :param input_file: the input file path
    :return: None
    """
    data = read_results_pickle(input_file)
    # print(data.prettyprint())
    link_loads_internal = data[0][1]["LINK_LOAD"]["PER_LINK_INTERNAL"]
    link_loads_external = data[0][1]["LINK_LOAD"]["PER_LINK_EXTERNAL"]
    # node loads is the sum of link loads which connect to the node
    node_loads = defaultdict(float)
    for link in link_loads_internal:
        ((u, v),), link_load = link
        # print("Internal link: {} -> {} : {}".format(u, v, link_load))
        node_loads[u] += link_load
        node_loads[v] += link_load
    for link in link_loads_external:
        ((u, v),), link_load = link
        node_loads[u] += link_load
        node_loads[v] += link_load
    # dump node loads and link loads to output file
    with open(output_file, "w") as f:
        for link in link_loads_internal:
            ((u, v),), link_load = link
            f.write(str(u) + " " + str(v) + " " + str(node_loads[u]) + " " + str(node_loads[v]) + " " + str(
                link_load) + "\n")
        for link in link_loads_external:
            ((u, v),), link_load = link
            f.write(str(u) + " " + str(v) + " " + str(node_loads[u]) + " " + str(node_loads[v]) + " " + str(
                link_load) + "\n")

    # print("Node loads: {}".format(node_loads))
    # print node max load and min load
    max_load = max(node_loads.values())
    min_load = min(node_loads.values())
    print("Max load: {} / min load: {}".format(max_load, min_load))


def csv_builder(input_file: str = "results.pickle", output_file: str = "result.csv"):
    """
    Build the csv file from the flow data file
    :param input_file: the input file path
    :param output_file: the output file path
    :return: None
    """
    # read the flow data file
    data = read_results_pickle(input_file)
    # print(data.prettyprint())

    # dump the data to csv file
    with open(output_file, "w") as f:
        f.write(
            "method,workload,mean_internal,mean_external,nsd_internal,nsd_external,avg_source_load,max_source_load,std_source_load\n")

        for i in range(len(data)):
            mean_internal = data[i][1]["LINK_LOAD"]["MEAN_INTERNAL"]
            mean_external = data[i][1]["LINK_LOAD"]["MEAN_EXTERNAL"]
            nsd_internal = data[i][1]["LINK_LOAD"]["NSD_INTERNAL"]
            nsd_external = data[i][1]["LINK_LOAD"]["NSD_EXTERNAL"]
            avg_source_load = data[i][1]["SOURCE_LOAD"]["AVG_SOURCE_LOAD"]
            max_source_load = data[i][1]["SOURCE_LOAD"]["MAX_SOURCE_LOAD"]
            std_source_load = data[i][1]["SOURCE_LOAD"]["STD_SOURCE_LOAD"]
            method = data[i][0]["strategy"]["method"]
            desc = data[i][0]["desc"]
            workload = desc.split(" / ")[-1].split(": ")[-1]
            f.write(method + "," + workload + "," + str(mean_internal) + "," + str(mean_external) + "," + str(
                nsd_internal) + "," + str(nsd_external) + "," + str(avg_source_load) + "," + str(
                max_source_load) + "," + str(std_source_load))
            f.write("\n")

    print("CSV builder, Done!")


if __name__ == '__main__':
    input_f, out_f = sys.argv[1], sys.argv[2]
    if ".csv" in out_f:
        csv_builder(input_f, out_f)
    else:
        dump_flow_data(input_f, out_f)
    print("Done!")
