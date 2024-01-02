"""Content placement strategies.

This module contains function to decide the allocation of content objects to
source nodes.
"""
import math
import random
import collections

from os import path
from fnss.util import random_from_pdf
from icarus.registry import register_content_placement

__all__ = ["uniform_content_placement", "weighted_content_placement", "redundant_content_placement"]
# Path where all workloads are stored
WORKLOAD_RESOURCES_DIR = path.abspath(
    path.join(
        path.dirname(__file__), path.pardir, path.pardir, "resources", "workloads"
    )
)


def apply_content_placement(placement, topology):
    """Apply a placement to a topology

    Parameters
    ----------
    placement : dict of sets
        Set of contents to be assigned to nodes keyed by node identifier
    topology : Topology
        The topology
    """
    for v, contents in placement.items():
        topology.node[v]["stack"][1]["contents"] = contents


def get_sources(topology):
    return [v for v in topology if topology.node[v]["stack"][0] == "source"]


@register_content_placement("UNIFORM")
def uniform_content_placement(topology, contents, seed=None):
    """Places content objects to source nodes randomly following a uniform
    distribution.

    Parameters
    ----------
    topology : Topology
        The topology object
    contents : Iterable of content objects
    seed : any hashable type, optional
        The seed to be used for random number generation

    Returns
    -------
    cache_placement : dict
        Dictionary mapping content objects to source nodes

    Notes
    -----
    A deterministic placement of objects (e.g., for reproducing results) can be
    achieved by using a fix seed value
    """
    random.seed(seed)
    source_nodes = get_sources(topology)
    content_placement = collections.defaultdict(set)
    for c in contents:
        content_placement[random.choice(source_nodes)].add(c)
    apply_content_placement(content_placement, topology)
    return content_placement


@register_content_placement("WEIGHTED")
def weighted_content_placement(topology, contents, source_weights, seed=None):
    """Places content objects to source nodes randomly according to the weight
     of the source node.

     Parameters
     ----------
     topology : Topology
         The topology object
     contents : iterable
         Iterable of content objects
     source_weights : dict
         Dict mapping nodes nodes of the topology which are content sources and
         the weight according to which content placement decision is made.

     Returns
     -------
     cache_placement : dict
         Dictionary mapping content objects to source nodes

     Notes
     -----
     A deterministic placement of objects (e.g., for reproducing results) can be
     achieved by using a fix seed value
    """
    random.seed(seed)
    norm_factor = float(sum(source_weights.values()))
    source_pdf = {k: v / norm_factor for k, v in source_weights.items()}
    content_placement = collections.defaultdict(set)
    for c in contents:
        content_placement[random_from_pdf(source_pdf)].add(c)
    apply_content_placement(content_placement, topology)
    return content_placement


@register_content_placement("REDUNDANT")
def redundant_content_placement(topology, contents, content_file, source_weights=None, seed=None):
    """
    In practical scenarios, a content will be cached by multiple source nodes.
    The strategy implemented here is redundant storage.
    Parameters
    ----------
    topology : Topology
         The topology object
    contents : iterable
            Iterable of content objects
    source_weights : dict
            Dict mapping nodes nodes of the topology which are content sources and
            the weight_link according to which content placement decision is made.
    content_file : file
            content csv file, (content, popularity, size, app_type)
    seed : any hashable type, optional
    """
    if source_weights is None:
        source_weights = {v: 1 for v in get_sources(topology)}
    norm_factor = float(sum(source_weights.values()))
    source_pdf = {k: v / norm_factor for k, v in source_weights.items()}
    # read content file
    content_pdf = {}
    if isinstance(content_file, dict):
        for content, popularity in content_file.items():
            content_pdf[content] = float(popularity)
    else:
        cf = WORKLOAD_RESOURCES_DIR + content_file
        with open(cf, 'r') as f:
            for line in f:
                content, popularity, size, app_type = line.split(',')
                content_pdf[int(content)] = float(popularity)
    # target popularity is the top alpha of the popularity, alpha default is 0.3
    target_pop = sorted(content_pdf.values(), reverse=True)[math.ceil(len(content_pdf) * 0.3)]
    # print("[redundant_content_placement] Target popularity: ", target_pop)

    content_placement = collections.defaultdict(set)

    # create a list of contents with redundancy, content number is proportional to its weight_link.
    # The maximum number depends on population. The minimum number of copies is 1.
    content_list = []
    for c in content_pdf:
        content_list.extend([c] * max(1, int(content_pdf[c] / target_pop)))
    # shuffle the content list
    # random.shuffle(content_list)
    # assign the content to source nodes
    # print(content_list)
    # choose any source node use source_pdf with seed
    random.seed(seed)
    for c in content_list:
        content_placement[random.choice(list(source_pdf.keys()))].add(c)
    # for c in content_list:
    #     content_placement[random_from_pdf(source_pdf)].add(c)
    apply_content_placement(content_placement, topology)
    return content_placement


if __name__ == '__main__':
    # test redundant_content_placement
    import fnss

    xtopology = fnss.k_ary_tree_topology(3, 3)
    xcontents = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    xsource_weights = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    xcontent_weights = {1: 6, 2: 0.8, 3: 0.05, 4: 0.02, 5: 0.03, 6: 10, 7: 20, 8: 0.01, 9: 0.01}
    cp = redundant_content_placement(xtopology, xcontents, xcontent_weights, xsource_weights)
    print(cp)
