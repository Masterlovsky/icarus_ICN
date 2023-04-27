"""Traffic workloads

Every traffic workload to be used with Icarus must be modelled as an iterable
class, i.e. a class with at least an `__init__` method (through which it is
initialized, with values taken from the configuration file) and an `__iter__`
method that is called to return a new event.

Each call to the `__iter__` method must return a 2-tuple in which the first
element is the timestamp at which the event occurs and the second is a
dictionary, describing the event, which must contain at least the three
following attributes:
 * receiver: The name of the node issuing the request
 * content: The name of the content for which the request is issued
 * log: A boolean value indicating whether this request should be logged or not
   for measurement purposes.

Each workload must expose the `contents` attribute which is an iterable of
all content identifiers. This is needed for content placement.
"""
import random
import csv
import collections

import fnss
import networkx as nx
from tqdm import tqdm
from os import path
import pandas as pd
from icarus.tools import TruncatedZipfDist
from icarus.registry import register_workload

__all__ = [
    "StationaryWorkload",
    "GlobetraffWorkload",
    "TraceDrivenWorkload",
    "YCSBWorkload",
    "LPWorkload",
    "REALWorkload",
]

# Path where all workloads are stored
WORKLOAD_RESOURCES_DIR = path.abspath(
    path.join(
        path.dirname(__file__), path.pardir, path.pardir, "resources", "workloads"
    )
)


def get_nodes_with_type(topology, ntype="receiver"):
    """
    Return the nodes of a topology, which are of a given type(receivers or source).
    """
    return [v for v in topology if topology.node[v]["stack"][0] == ntype]


def get_contents(topology, v):
    """
    Return the contents in a source node.
    """
    return topology.node[v]["stack"][1].get("contents", set())


@register_workload("STATIONARY")
class StationaryWorkload:
    """This function generates events on the fly, i.e. instead of creating an
    event schedule to be kept in memory, returns an iterator that generates
    events when needed.

    This is useful for running large schedules of events where RAM is limited
    as its memory impact is considerably lower.

    These requests are Poisson-distributed while content popularity is
    Zipf-distributed

    All requests are mapped to receivers uniformly unless a positive *beta*
    parameter is specified.

    If a *beta* parameter is specified, then receivers issue requests at
    different rates. The algorithm used to determine the requests rates for
    each receiver is the following:
     * All receiver are sorted in decreasing order of degree of the PoP they
       are attached to. This assumes that all receivers have degree = 1 and are
       attached to a node with degree > 1
     * Rates are then assigned following a Zipf distribution of coefficient
       beta where nodes with higher-degree PoPs have a higher request rate

    Parameters
    ----------
    topology : fnss.Topology
        The topology to which the workload refers
    n_contents : int
        The number of content object
    alpha : float
        The Zipf alpha parameter
    beta : float, optional
        Parameter indicating
    rate : float, optional
        The mean rate of requests per second
    n_warmup : int, optional
        The number of warmup requests (i.e. requests executed to fill cache but
        not logged)
    n_measured : int, optional
        The number of logged requests after the warmup

    Returns
    -------
    events : iterator
        Iterator of events. Each event is a 2-tuple where the first element is
        the timestamp at which the event occurs and the second element is a
        dictionary of event attributes.
    """

    def __init__(
            self,
            topology,
            n_contents,
            alpha,
            beta=0,
            rate=1.0,
            n_warmup=10 ** 5,
            n_measured=4 * 10 ** 5,
            seed=None,
            **kwargs
    ):
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if beta < 0:
            raise ValueError("beta must be positive")
        self.topology = topology
        self.sources = get_nodes_with_type(topology, "source")
        self.receivers = get_nodes_with_type(topology, "receiver")
        self.zipf = TruncatedZipfDist(alpha, n_contents)
        self.n_contents = n_contents
        self.contents = range(1, n_contents + 1)
        self.alpha = alpha
        self.rate = rate
        self.n_warmup = n_warmup
        self.n_measured = n_measured
        random.seed(seed)
        self.beta = beta
        if beta != 0:
            degree = nx.degree(self.topology)
            self.receivers = sorted(
                self.receivers,
                key=lambda x: degree[iter(topology.adj[x]).__next__()],
                reverse=True,
            )
            self.receiver_dist = TruncatedZipfDist(beta, len(self.receivers))

    def __iter__(self):
        req_counter = 0
        t_event = 0.0
        while req_counter < self.n_warmup + self.n_measured:
            t_event += random.expovariate(self.rate)
            if self.beta == 0:
                receiver = random.choice(self.receivers)
            else:
                receiver = self.receivers[self.receiver_dist.rv() - 1]
            content = int(self.zipf.rv())
            log = req_counter >= self.n_warmup
            event = {"receiver": receiver, "content": content, "log": log}
            yield (t_event, event)
            req_counter += 1
        return


@register_workload("GLOBETRAFF")
class GlobetraffWorkload:
    """Parse requests from GlobeTraff workload generator

    All requests are mapped to receivers uniformly unless a positive *beta*
    parameter is specified.

    If a *beta* parameter is specified, then receivers issue requests at
    different rates. The algorithm used to determine the requests rates for
    each receiver is the following:
     * All receiver are sorted in decreasing order of degree of the PoP they
       are attached to. This assumes that all receivers have degree = 1 and are
       attached to a node with degree > 1
     * Rates are then assigned following a Zipf distribution of coefficient
       beta where nodes with higher-degree PoPs have a higher request rate

    Parameters
    ----------
    topology : fnss.Topology
        The topology to which the workload refers
    reqs_file : str
        The GlobeTraff request file
    contents_file : str
        The GlobeTraff content file
    beta : float, optional
        Spatial skewness of requests rates

    Returns
    -------
    events : iterator
        Iterator of events. Each event is a 2-tuple where the first element is
        the timestamp at which the event occurs and the second element is a
        dictionary of event attributes.
    """

    def __init__(self, topology, reqs_file, contents_file, beta=0, **kwargs):
        """Constructor"""
        if beta < 0:
            raise ValueError("beta must be positive")
        self.receivers = get_nodes_with_type(topology, "receiver")
        self.n_contents = 0
        with open(contents_file) as f:
            reader = csv.reader(f, delimiter="\t")
            for content, popularity, size, app_type in reader:
                self.n_contents = max(self.n_contents, content)
        self.n_contents += 1
        self.contents = range(self.n_contents)
        self.request_file = reqs_file
        self.beta = beta
        if beta != 0:
            degree = nx.degree(self.topology)
            self.receivers = sorted(
                self.receivers,
                key=lambda x: degree[iter(topology.adj[x]).next()],
                reverse=True,
            )
            self.receiver_dist = TruncatedZipfDist(beta, len(self.receivers))

    def __iter__(self):
        with open(self.request_file) as f:
            reader = csv.reader(f, delimiter="\t")
            for timestamp, content, size in reader:
                if self.beta == 0:
                    receiver = random.choice(self.receivers)
                else:
                    receiver = self.receivers[self.receiver_dist.rv() - 1]
                event = {"receiver": receiver, "content": content, "size": size}
                yield (timestamp, event)
        return


@register_workload("TRACE_DRIVEN")
class TraceDrivenWorkload:
    """Parse requests from a generic request trace.

    This workload requires two text files:
     * a requests file, where each line corresponds to a string identifying
       the content requested
     * a contents file, which lists all unique content identifiers appearing
       in the requests file.

    Since the trace do not provide timestamps, requests are scheduled according
    to a Poisson process of rate *rate*. All requests are mapped to receivers
    uniformly unless a positive *beta* parameter is specified.

    If a *beta* parameter is specified, then receivers issue requests at
    different rates. The algorithm used to determine the requests rates for
    each receiver is the following:
     * All receiver are sorted in decreasing order of degree of the PoP they
       are attached to. This assumes that all receivers have degree = 1 and are
       attached to a node with degree > 1
     * Rates are then assigned following a Zipf distribution of coefficient
       beta where nodes with higher-degree PoPs have a higher request rate

    Parameters
    ----------
    topology : fnss.Topology
        The topology to which the workload refers
    reqs_file : str
        The path to the requests file
    contents_file : str
        The path to the contents file
    n_contents : int
        The number of content object (i.e. the number of lines of contents_file)
    n_warmup : int
        The number of warmup requests (i.e. requests executed to fill cache but
        not logged)
    n_measured : int
        The number of logged requests after the warmup
    rate : float, optional
        The network-wide mean rate of requests per second
    beta : float, optional
        Spatial skewness of requests rates

    Returns
    -------
    events : iterator
        Iterator of events. Each event is a 2-tuple where the first element is
        the timestamp at which the event occurs and the second element is a
        dictionary of event attributes.
    """

    def __init__(
            self,
            topology,
            reqs_file,
            contents_file,
            n_contents,
            n_warmup,
            n_measured,
            rate=1.0,
            beta=0,
            **kwargs
    ):
        """Constructor"""
        if beta < 0:
            raise ValueError("beta must be positive")
        # Set high buffering to avoid one-line reads
        self.buffering = 64 * 1024 * 1024
        self.n_contents = n_contents
        self.n_warmup = n_warmup
        self.n_measured = n_measured
        self.reqs_file = reqs_file
        self.rate = rate
        self.receivers = [
            v for v in topology.nodes() if topology.node[v]["stack"][0] == "receiver"
        ]
        self.contents = []
        with open(contents_file, buffering=self.buffering) as f:
            for content in f:
                self.contents.append(content)
        self.beta = beta
        if beta != 0:
            degree = nx.degree(topology)
            self.receivers = sorted(
                self.receivers,
                key=lambda x: degree[iter(topology.adj[x]).next()],
                reverse=True,
            )
            self.receiver_dist = TruncatedZipfDist(beta, len(self.receivers))

    def __iter__(self):
        req_counter = 0
        t_event = 0.0
        with open(self.reqs_file, buffering=self.buffering) as f:
            for content in f:
                t_event += random.expovariate(self.rate)
                if self.beta == 0:
                    receiver = random.choice(self.receivers)
                else:
                    receiver = self.receivers[self.receiver_dist.rv() - 1]
                log = req_counter >= self.n_warmup
                event = {"receiver": receiver, "content": content, "log": log}
                yield (t_event, event)
                req_counter += 1
                if req_counter >= self.n_warmup + self.n_measured:
                    return
            raise ValueError("Trace did not contain enough requests")


@register_workload("YCSB")
class YCSBWorkload:
    """Yahoo! Cloud Serving Benchmark (YCSB)

    The YCSB is a set of reference workloads used to benchmark databases and,
    more generally any storage/caching systems. It comprises five workloads:

    +------------------+------------------------+------------------+
    | Workload         | Operations             | Record selection |
    +------------------+------------------------+------------------+
    | A - Update heavy | Read: 50%, Update: 50% | Zipfian          |
    | B - Read heavy   | Read: 95%, Update: 5%  | Zipfian          |
    | C - Read only    | Read: 100%             | Zipfian          |
    | D - Read latest  | Read: 95%, Insert: 5%  | Latest           |
    | E - Short ranges | Scan: 95%, Insert 5%   | Zipfian/Uniform  |
    +------------------+------------------------+------------------+

    Notes
    -----
    At the moment only workloads A, B and C are implemented, since they are the
    most relevant for caching systems.
    """

    def __init__(
            self,
            workload,
            n_contents,
            n_warmup,
            n_measured,
            alpha=0.99,
            seed=None,
            **kwargs
    ):
        """Constructor

        Parameters
        ----------
        workload : str
            Workload identifier. Currently supported: "A", "B", "C"
        n_contents : int
            Number of content items
        n_warmup : int, optional
            The number of warmup requests (i.e. requests executed to fill cache but
            not logged)
        n_measured : int, optional
            The number of logged requests after the warmup
        alpha : float, optional
            Parameter of Zipf distribution
        seed : int, optional
            The seed for the random generator
        """

        if workload not in ("A", "B", "C", "D", "E"):
            raise ValueError("Incorrect workload ID [A-B-C-D-E]")
        elif workload in ("D", "E"):
            raise NotImplementedError("Workloads D and E not yet implemented")
        self.workload = workload
        if seed is not None:
            random.seed(seed)
        self.zipf = TruncatedZipfDist(alpha, n_contents)
        self.n_warmup = n_warmup
        self.n_measured = n_measured

    def __iter__(self):
        """Return an iterator over the workload"""
        req_counter = 0
        while req_counter < self.n_warmup + self.n_measured:
            rand = random.random()
            op = {
                "A": "READ" if rand < 0.5 else "UPDATE",
                "B": "READ" if rand < 0.95 else "UPDATE",
                "C": "READ",
            }[self.workload]
            item = int(self.zipf.rv())
            log = req_counter >= self.n_warmup
            event = {"op": op, "item": item, "log": log}
            yield event
            req_counter += 1
        return


@register_workload("LEVEL_PROBABILITY")
class LPWorkload:
    """This function generates events on the fly, i.e. instead of creating an
    event schedule to be kept in memory, returns an iterator that generates
    events when needed.

    This is useful for running large schedules of events where RAM is limited
    as its memory impact is considerably lower.

    These requests are Poisson-distributed.
    Depends on the neighborhood effect, the probability that a user requests an item
    is correlated with the content in local source. We call the probability that a user
    requests an item in local source as the level probability -- LP.

    All requests are mapped to receivers uniformly unless a positive *beta*
    parameter is specified.

    If a *beta* parameter is specified, then receivers issue requests at
    different rates. The algorithm used to determine the requests rates for
    each receiver is the following:
     * All receiver are sorted in decreasing order of degree of the PoP they
       are attached to. This assumes that all receivers have degree = 1 and are
       attached to a node with degree > 1
     * Rates are then assigned following a Zipf distribution of coefficient
       beta where nodes with higher-degree PoPs have a higher request rate

    Parameters
    ----------
    topology : fnss.Topology
        The topology to which the workload refers
    n_contents : int
        The number of content object
    alpha : float
        The Zipf alpha parameter
    lp : float
        The level probability
    beta : float, optional
        Parameter indicating
    rate : float, optional
        The mean rate of requests per second
    n_warmup : int, optional
        The number of warmup requests (i.e. requests executed to fill cache but
        not logged)
    n_measured : int, optional
        The number of logged requests after the warmup

    Returns
    -------
    events : iterator
        Iterator of events. Each event is a 2-tuple where the first element is
        the timestamp at which the event occurs and the second element is a
        dictionary of event attributes.
    """

    def __init__(
            self,
            topology,
            n_contents,
            alpha,
            lp,
            beta=0,
            rate=1.0,
            n_warmup=10 ** 5,
            n_measured=4 * 10 ** 5,
            seed=None,
            **kwargs
    ):
        if not 0 < lp < 1:
            raise ValueError("lp must be in (0, 1)")
        if beta < 0:
            raise ValueError("beta must be positive")
        self.topology = topology
        self.sources = get_nodes_with_type(topology, "source")
        self.receivers = get_nodes_with_type(topology, "receiver")
        self.n_contents = n_contents
        self.contents = range(1, n_contents + 1)
        self.lp = lp
        self.rate = rate
        self.n_warmup = n_warmup
        self.n_measured = n_measured
        random.seed(seed)
        self.beta = beta
        if beta != 0:
            degree = nx.degree(self.topology)
            self.receivers = sorted(
                self.receivers,
                key=lambda x: degree[iter(topology.adj[x]).__next__()],
                reverse=True,
            )
            self.receiver_dist = TruncatedZipfDist(beta, len(self.receivers))

        #  create a dict to store each level's zipf distribution for each receiver.
        self.zipf_dict = collections.defaultdict(list)
        self.receiver2contents_dict = collections.defaultdict(list)
        for receiver in tqdm(self.receivers, desc="Creating zipf distributions for each receiver: "):
            cont_ctrl, cont_mng = self.get_same_area_contents(receiver)
            cont_inter = set(self.contents) - cont_ctrl - cont_mng
            self.receiver2contents_dict[receiver] = [list(cont_ctrl), list(cont_mng), list(cont_inter)]
            zipf_ctrl = TruncatedZipfDist(alpha, len(cont_ctrl)) if cont_ctrl else None
            zipf_mng = TruncatedZipfDist(alpha, len(cont_mng)) if cont_mng else None
            zipf_inter = TruncatedZipfDist(alpha, len(cont_inter)) if cont_inter else None
            self.zipf_dict[receiver] = [zipf_ctrl, zipf_mng, zipf_inter]

    def __iter__(self):
        req_counter = 0
        t_event = 0.0

        while req_counter < self.n_warmup + self.n_measured:
            t_event += random.expovariate(self.rate)
            if self.beta == 0:
                receiver = random.choice(self.receivers)
            else:
                receiver = self.receivers[self.receiver_dist.rv() - 1]
            # generate a random value in [0, 1]
            p = random.random()
            if p < self.lp:
                contents = self.receiver2contents_dict[receiver][0]
                if not contents:
                    continue
                content = contents[self.zipf_dict[receiver][0].rv() - 1]
            elif p < self.lp + self.lp * (1 - self.lp):
                contents = self.receiver2contents_dict[receiver][1]
                if not contents:
                    continue
                content = self.receiver2contents_dict[receiver][1][self.zipf_dict[receiver][1].rv() - 1]
            else:
                content = self.receiver2contents_dict[receiver][2][self.zipf_dict[receiver][2].rv() - 1]
            log = req_counter >= self.n_warmup
            event = {"receiver": receiver, "content": content, "log": log}
            yield (t_event, event)
            req_counter += 1
        return

    def get_same_area_contents(self, v):
        """
        Get all content in the same control domain and management domain with the target node.
        :param v: target node
        :return: (set, set): contents in the same control domain, contents in the same management domain
        """
        contents_ctrl = set()
        contents_manage = set()
        ctrl_domain_v = self.topology.node[v]["stack"][1]["ctrl"]
        asn_v = self.topology.node[v]["stack"][1]["asn"]
        for u in self.topology.nodes():
            if self.topology.node[u]["stack"][0] != "source":
                continue
            if self.topology.node[u]["stack"][1]["asn"] == asn_v:
                if self.topology.node[u]["stack"][1]["ctrl"] == ctrl_domain_v:
                    contents_ctrl |= get_contents(self.topology, u)
                else:
                    contents_manage |= get_contents(self.topology, u)
        return contents_ctrl, contents_manage


@register_workload("REAL")
class REALWorkload(object):
    """Parse requests from Real log trace workload generator

    All requests are mapped to receivers uniformly unless a positive *beta*
    parameter is specified.

    Parameters
    ----------
    topology : fnss.Topology
        The topology to which the workload refers
    reqs_file : str
        The real workload trace request file

    Returns
    -------
    events : iterator
        Iterator of events. Each event is a 2-tuple where the first element is
        the timestamp at which the event occurs and the second element is a
        dictionary of event attributes.
    """

    def __init__(self, topology, reqs_file, summarize_file, **kwargs):
        self.topology = topology
        self.receivers = get_nodes_with_type(topology, "receiver")
        self.routers = get_nodes_with_type(topology, "router")
        self.reqs_f = open(WORKLOAD_RESOURCES_DIR + reqs_file, "r")
        self.reqs_df = pd.read_csv(self.reqs_f, sep=",")
        # read the first line of summarize_file to get the number of contents
        self.summarize_file = open(WORKLOAD_RESOURCES_DIR + summarize_file, "r")
        self.n_contents = int(self.summarize_file.readline().split(":")[1])
        self.city_num = int(self.summarize_file.readline().split(":")[1])
        self.summarize_file.readline()  # skip the line of "total rows"
        self.content_popularity = eval(self.summarize_file.readline().split(":")[1])
        self.contents = range(1, self.n_contents + 1)
        self.router2recv = collections.defaultdict(list)
        for router in self.routers:
            # get the receivers connected to the router
            for receiver in self.receivers:
                if receiver in topology.adj[router]:
                    self.router2recv[router].append(receiver)

    def __iter__(self):
        req_counter = 0
        t_event = 0.0
        for index, (timestamp, client, sw, content, size) in self.reqs_df.iterrows():
            t_event = float(timestamp)
            # get receiver by sw, receiver should be a node connected to sw
            if int(sw) not in self.routers:
                continue
            receiver = random.choice(self.router2recv[int(sw)])
            event = {"receiver": receiver, "content": int(content), "log": True, "size": int(size), "index": index}
            req_counter += 1
            yield t_event, event
        self.reqs_f.close()
        self.summarize_file.close()
        return
