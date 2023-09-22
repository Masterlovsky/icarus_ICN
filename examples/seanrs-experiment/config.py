"""Configuration file for running a single simple simulation."""
from multiprocessing import cpu_count
from collections import deque
import copy
from icarus.util import Tree

# GENERAL SETTINGS

# Level of logging output
# Available options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# If True, executes simulations in parallel using multiple processes
# to take advantage of multicore CPUs
PARALLEL_EXECUTION = True

# Number of processes used to run simulations in parallel.
# This option is ignored if PARALLEL_EXECUTION = False
N_PROCESSES = cpu_count()

# Number of times each experiment is replicated
N_REPLICATIONS = 1

# Granularity of caching.
# Currently, only OBJECT is supported
CACHING_GRANULARITY = "OBJECT"

# Format in which results are saved.
# Result readers and writers are located in module ./icarus/results/readwrite.py
# Currently only PICKLE is supported
RESULTS_FORMAT = "PICKLE"

# List of metrics to be measured in the experiments
# The implementation of data collectors are located in ./icarus/execution/collectors.py
DATA_COLLECTORS = ["CACHE_HIT_RATIO", "LINK_LOAD", "LATENCY", "PACKET_IN", "LEVEL_HIT"]
# DATA_COLLECTORS = ["LINK_LOAD"]

# Number of requests per second (over the whole network)
REQ_RATE = 10**8
# ALPHA = [0.6, 0.8, 1.0]
# Queue of experiments
EXPERIMENT_QUEUE = deque()

# Create experiment
experiment = Tree()

# Set topology
# experiment["topology"]["name"] = "SEANRS_SIMPLE"
experiment["topology"]["name"] = "SEANRS"
experiment["topology"]["scale"] = "DINNRS_topo"

# Set workload
experiment["workload"] = {
    "name": "LEVEL_PROBABILITY",  # LEVEL_PROBABILITY, STATIONARY
    "n_contents": 10 ** 4,
    "n_warmup": 5 * 10 ** 4,
    "n_measured": 5 * 10 ** 4,
    "alpha": 0.85,
    "lp": 0.8,
    "rate": REQ_RATE,
    # "seed": 1234,
}

# Set cache placement
experiment["cache_placement"]["name"] = "UNIFORM"
experiment["cache_placement"]["network_cache"] = 10.0

# Set content placement
experiment["content_placement"]["name"] = "UNIFORM"

# Set cache replacement policy
experiment["cache_policy"]["name"] = "LRU"

# Set caching meta-policy
experiment["strategy"]["name"] = "SEANRS"

# Description of the experiment
experiment["desc"] = "SEANRS simple topology test"

# Append experiment to queue
# EXPERIMENT_QUEUE.append(experiment)

for workload in ("LEVEL_PROBABILITY", "STATIONARY"):
    for network_cache in ([10 ** i for i in range(-2, 3)]):
        for req_rate in range(500, 1400, 100):
            if workload == "LEVEL_PROBABILITY":
                for lp in (0.3, 0.5, 0.7, 0.9):
                    extra_experiment = copy.deepcopy(experiment)
                    extra_experiment["workload"]["name"] = workload
                    extra_experiment["cache_placement"]["network_cache"] = network_cache
                    extra_experiment["workload"]["rate"] = req_rate
                    extra_experiment["workload"]["lp"] = lp
                    extra_experiment["desc"] = ("DINNRS / workload: {} / network_cache: {} / req_rate: {} / lp: {}"
                                                .format(workload, network_cache, req_rate, lp))
                    EXPERIMENT_QUEUE.append(extra_experiment)
            else:
                extra_experiment = copy.deepcopy(experiment)
                extra_experiment["workload"]["name"] = workload
                extra_experiment["cache_placement"]["network_cache"] = network_cache
                extra_experiment["workload"]["rate"] = req_rate
                extra_experiment["desc"] = ("DINNRS / workload: {} / network_cache: {} / req_rate: {}"
                                            .format(workload, network_cache, req_rate))
                EXPERIMENT_QUEUE.append(extra_experiment)
