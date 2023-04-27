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
PARALLEL_EXECUTION = False

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
DATA_COLLECTORS = ["CACHE_HIT_RATIO"]
# DATA_COLLECTORS = ["LINK_LOAD"]

# Number of requests per second (over the whole network)
REQ_RATE = 10
# ALPHA = [0.6, 0.8, 1.0]
# Queue of experiments
EXPERIMENT_QUEUE = deque()

# Create experiment
experiment = Tree()

# Set topology
experiment["topology"]["name"] = "TREE"
experiment["topology"]["k"] = 10
experiment["topology"]["h"] = 2

# # Set workload
# experiment["workload"] = {
#     "name": "STATIONARY",
#     "n_contents": 10 ** 3,
#     "n_warmup": 5 * 10 ** 3,
#     "n_measured": 5 * 10 ** 3,
#     "alpha": 0.85,
#     # "lp": 0.8,
#     "rate": REQ_RATE,
#     # "seed": 1234,
# }
# Set Real Workload
experiment["workload"] = {
    "name": "REAL",
    "reqs_file": "/20220610/df.csv",
    "summarize_file": "/20220610/summarize.txt",
    "n_contents": 26240,
}

# Set cache placement
experiment["cache_placement"]["name"] = "UNIFORM"
experiment["cache_placement"]["network_cache"] = 0.1

# Set content placement
experiment["content_placement"]["name"] = "UNIFORM"

# Set cache replacement policy
experiment["cache_policy"]["name"] = "LRU"
experiment["cache_policy"]["timeout"] = True
experiment["cache_policy"]["t0"] = 1000

# Set caching meta-policy
experiment["strategy"]["name"] = "SEACACHE"

# Description of the experiment
experiment["desc"] = "SEACACHE simple topology test"

# Append experiment to queue
EXPERIMENT_QUEUE.append(experiment)