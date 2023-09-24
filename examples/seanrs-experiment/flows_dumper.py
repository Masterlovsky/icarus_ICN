#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Script to dump the flows data from result files
Created on 2022-11-11, powered by Masterlovsky
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


def dump_result_to_csv(input_file: str = "result.pickle", out_file: str = "result.csv"):
    """
    Dump the result to csv file
    """
    data = read_results_pickle(input_file)
    with open(out_file, "w") as f:
        # write title
        f.write(
            "exp_id,topology_scale,workload_name,workload_n_contents,alpha,lp,rate,cache,"
            "strategy,avg_chr,intra_link_load,inter_link_load,avg_latency,pktin_total,"
            "pktin_cpr,req_num,resolve_cache,resolve_ctrl,resolve_ibgn,"
            "resolve_ebgn,avg_cc_cache,avg_cc_ctrl,avg_cc_ibgn,avg_cc_ebgn\n"
        )
        exp_numbers = len(data)
        for i in range(exp_numbers):
            # experiment config data
            exp_id = str(i)
            topology_scale = data[i][0]["topology"]["scale"]
            workload_name = data[i][0]["workload"]["name"]
            workload_n_contents = str(data[i][0]["workload"]["n_contents"])
            alpha = str(data[i][0]["workload"]["alpha"])
            lp = "NULL" if "lp" not in data[i][0]["workload"] else str(data[i][0]["workload"]["lp"])
            rate = str(data[i][0]["workload"]["rate"])
            cache = str(data[i][0]["cache_placement"]["network_cache"])
            strategy = data[i][0]["strategy"]["name"]
            # result data
            avg_chr = "NULL" if ("CACHE_HIT_RATIO" not in data[i][1] or "MEAN" not in data[i][1]["CACHE_HIT_RATIO"]) else str(
                data[i][1]["CACHE_HIT_RATIO"]["MEAN"])
            intra_link_load = "NULL" if "LINK_LOAD" not in data[i][1] else str(
                data[i][1]["LINK_LOAD"]["MEAN_INTERNAL"])
            inter_link_load = "NULL" if "LINK_LOAD" not in data[i][1] else str(
                data[i][1]["LINK_LOAD"]["MEAN_EXTERNAL"])
            avg_latency = "NULL" if "LATENCY" not in data[i][1] else str(data[i][1]["LATENCY"]["MEAN"])
            pktin_total = "NULL" if "PACKET_IN" not in data[i][1] else str(
                data[i][1]["PACKET_IN"]["PACKET_IN_COUNT_TOTAL"])
            pktin_cpr = "NULL" if "PACKET_IN" not in data[i][1] else str(
                data[i][1]["PACKET_IN"]["PACKET_IN_COUNT_MEAN"])
            req_num = "NULL" if "LEVEL_HIT" not in data[i][1] else str(data[i][1]["LEVEL_HIT"]["TOTAL_REQUEST"])
            resolve_cache = "NULL" if (
                    "LEVEL_HIT" not in data[i][1] or "RESOLVE_CACHE" not in data[i][1]["LEVEL_HIT"]) else str(
                data[i][1]["LEVEL_HIT"]["RESOLVE_CACHE"])
            resolve_ctrl, resolve_ibgn, resolve_ebgn = "NULL", "NULL", "NULL"
            if "LEVEL_HIT" in data[i][1]:
                if "RESOLVE_CTRL" in data[i][1]["LEVEL_HIT"]:
                    resolve_ctrl = str(data[i][1]["LEVEL_HIT"]["RESOLVE_CTRL"])
                elif "RESOLVE_L1" in data[i][1]["LEVEL_HIT"]:
                    resolve_ctrl = str(data[i][1]["LEVEL_HIT"]["RESOLVE_L1"])
                if "RESOLVE_IBGN" in data[i][1]["LEVEL_HIT"]:
                    resolve_ibgn = str(data[i][1]["LEVEL_HIT"]["RESOLVE_IBGN"])
                elif "RESOLVE_L2" in data[i][1]["LEVEL_HIT"]:
                    resolve_ibgn = str(data[i][1]["LEVEL_HIT"]["RESOLVE_L2"])
                if "RESOLVE_EBGN" in data[i][1]["LEVEL_HIT"]:
                    resolve_ebgn = str(data[i][1]["LEVEL_HIT"]["RESOLVE_EBGN"])
                elif "RESOLVE_L3" in data[i][1]["LEVEL_HIT"]:
                    resolve_ebgn = str(data[i][1]["LEVEL_HIT"]["RESOLVE_L3"])
            avg_cc_cache = "NULL" if (
                    "LEVEL_HIT" not in data[i][1] or "CONCURRENCY_CACHE_MEAN" not in data[i][1]["LEVEL_HIT"]) else str(
                data[i][1]["LEVEL_HIT"]["CONCURRENCY_CACHE_MEAN"])
            avg_cc_ctrl = "NULL" if (
                    "LEVEL_HIT" not in data[i][1] or "CONCURRENCY_CTRL_MEAN" not in data[i][1]["LEVEL_HIT"]) else str(
                data[i][1]["LEVEL_HIT"]["CONCURRENCY_CTRL_MEAN"])
            avg_cc_ibgn = "NULL" if (
                    "LEVEL_HIT" not in data[i][1] or "CONCURRENCY_IBGN_MEAN" not in data[i][1]["LEVEL_HIT"]) else str(
                data[i][1]["LEVEL_HIT"]["CONCURRENCY_IBGN_MEAN"])
            avg_cc_ebgn = "NULL" if (
                    "LEVEL_HIT" not in data[i][1] or "CONCURRENCY_EBGN_MEAN" not in data[i][1]["LEVEL_HIT"]) else str(
                data[i][1]["LEVEL_HIT"]["CONCURRENCY_EBGN_MEAN"])
            # write data to csv file
            f.write(
                exp_id + "," + topology_scale + "," + workload_name + "," + workload_n_contents + ","
                + alpha + "," + lp + "," + rate + "," + cache + "," + strategy + "," + avg_chr + "," +
                intra_link_load + "," + inter_link_load + "," + avg_latency + "," + pktin_total + "," +
                pktin_cpr + "," + req_num + "," + resolve_cache + "," + resolve_ctrl + "," + resolve_ibgn +
                "," + resolve_ebgn + "," + avg_cc_cache + "," + avg_cc_ctrl + "," + avg_cc_ibgn + "," + avg_cc_ebgn + "\n"
            )

    print("Dump result to csv done! Check {}".format(out_file))


if __name__ == '__main__':
    path = "./"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    dump_flow_data(path + "results.pickle", path + "flow_data.txt")
    dump_result_to_csv(path + "results.pickle", path + "result.csv")
    print("Done!")
