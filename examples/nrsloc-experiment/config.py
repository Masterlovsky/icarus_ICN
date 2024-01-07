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
# DATA_COLLECTORS = ["CACHE_HIT_RATIO", "FREE_SPACE"]
DATA_COLLECTORS = ["LINK_LOAD", "SOURCE_LOAD"]

# Number of requests per second (over the whole network)
REQ_RATE = 10
# ALPHA = [0.6, 0.8, 1.0]
# Queue of experiments
EXPERIMENT_QUEUE = deque()

# Create experiment
experiment = Tree()

# Set topology
experiment["topology"]["name"] = "TISCALI_SEA"

# Set workload
experiment["workload"] = {
    "name": "GLOBETRAFF",
    "n_contents": 1000,
    "reqs_file": "/P3/darknet_processed/req_file.csv",
    "content_file": "/P3/darknet_processed/content_file.csv",
}

# Set cache placement
experiment["cache_placement"]["name"] = "UNIFORM"
experiment["cache_placement"]["network_cache"] = 0.1

# Set content placement
experiment["content_placement"]["name"] = "REDUNDANT"
experiment["content_placement"]["content_file"] = "/P3/darknet_processed/content_file.csv"
experiment["content_placement"]["seed"] = 2024

# Set cache replacement policy
experiment["cache_policy"]["name"] = "LRU"
experiment["cache_policy"]["timeout"] = False
experiment["cache_policy"]["t0"] = 10 * 60

# Set caching meta-policy
# experiment["strategy"]["name"] = "LCE"
experiment["strategy"]["name"] = "SEALOC"
experiment["strategy"]["method"] = "main"  # {"main", "first", "onlystates", "onlyfreq", "random"}

# Description of the experiment
experiment["desc"] = "SEALOC / method: main / workload: darknet_processed"

# Append experiment to queue
EXPERIMENT_QUEUE.append(experiment)

# Copy experiment, change parameters and append to queue
# --------- experiment 1: different method ---------
for method in ("main", "first", "onlystates", "onlyfreq", "random"):
    for workload in ("darknet_processed", "sdn_processed", "bsy_processed"):
        for seed in range(2024, 2024 + 5):
            extra_experiment = copy.deepcopy(experiment)
            extra_experiment["workload"]["reqs_file"] = "/P3/{}/req_file.csv".format(workload)
            extra_experiment["workload"]["content_file"] = "/P3/{}/content_file.csv".format(workload)
            extra_experiment["content_placement"]["content_file"] = "/P3/{}/content_file.csv".format(workload)
            extra_experiment["content_placement"]["seed"] = seed
            extra_experiment["strategy"]["method"] = method
            extra_experiment["desc"] = "SEALOC / method: {} / workload: {}".format(method, workload)
            EXPERIMENT_QUEUE.append(extra_experiment)

