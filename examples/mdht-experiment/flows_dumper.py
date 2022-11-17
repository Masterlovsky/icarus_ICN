#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Script to dump the flows data from result files
Created on 2022-11-11, powered by Masterlovsky
Version: 1.0
"""
from collections import defaultdict

from icarus.results.readwrite import *


def dump_flow_data(input_file: str = "results.pickle", output_file: str = "flow_data.txt"):
    """
    Dump the flows data from result files
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


if __name__ == '__main__':
    path = "./"
    dump_flow_data(path + "results.pickle", path + "flow_data.txt")
    print("Done!")
