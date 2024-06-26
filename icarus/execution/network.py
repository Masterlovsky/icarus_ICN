"""Network Model-View-Controller (MVC)

This module contains classes providing an abstraction of the network shown to
the strategy implementation. The network is modelled using an MVC design
pattern.

A strategy performs actions on the network by calling methods of the
`NetworkController`, that in turns updates  the `NetworkModel` instance that
updates the `NetworkView` instance. The strategy can get updated information
about the network status by calling methods of the `NetworkView` instance.

The `NetworkController` is also responsible to notify a `DataCollectorProxy`
of all relevant events.
"""
import bisect
import logging
from collections import defaultdict
from sortedcontainers import SortedList
import fnss
import mmh3
import networkx as nx
from tqdm import tqdm

from icarus.execution.simulation_time import *
from icarus.models.cache.policies import ttl_cache
from icarus.registry import CACHE_POLICY
from icarus.util import iround, path_links
from icarus.utils.cuckoo import *
from icarus.utils.dht_chord import DHT, NodeDHT
from icarus.utils.uhashring import HashRing

__all__ = ["NetworkModel", "NetworkView", "NetworkController"]

logger = logging.getLogger("orchestration")


def symmetrify_paths(shortest_paths):
    """Make paths symmetric

    Given a dictionary of all-pair shortest paths, it edits shortest paths to
    ensure that all path are symmetric, e.g., path(u,v) = path(v,u)

    Parameters
    ----------
    shortest_paths : dict of dict
        All pairs shortest paths

    Returns
    -------
    shortest_paths : dict of dict
        All pairs shortest paths, with all paths symmetric

    Notes
    -----
    This function modifies the shortest paths dictionary provided
    """
    for u in shortest_paths:
        for v in shortest_paths[u]:
            shortest_paths[u][v] = list(reversed(shortest_paths[v][u]))
    return shortest_paths


class NetworkView:
    """Network view

    This class provides an interface that strategies and data collectors can
    use to know updated information about the status of the network.
    For example the network view provides information about shortest paths,
    characteristics of links and currently cached objects in nodes.
    """
    POP_RANGE = 10  # ! set 10s as a popularity calculate range

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            The network model instance
        """
        if not (isinstance(model, NetworkModel) or isinstance(model, SEANRSModel) or isinstance(model, MDHTModel)):
            raise ValueError(
                "The model argument must be an instance of " "NetworkModel"
            )
        self.model = model

    def content_locations(self, k):
        """Return a set of all current locations of a specific content.

        This include both persistent content sources and temporary caches.

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        nodes : set
            A set of all nodes currently storing the given content
        """
        loc = {v for v in self.model.cache if self.model.cache[v].has(k)}
        source = self.model.content_source.get(k)  # a list of source node
        if source:
            loc.update(source)
        return loc

    def content_source(self, k):
        """Return the node identifier where the content is persistently stored.

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        node : any hashable type
            The node persistently storing the given content or None if the
            source is unavailable
        """
        src = self.model.content_source.get(k)
        return src[0] if src else None

    def get_access_switch(self, v):
        """
        ---- Only used in SEANet topology. ----
        Return the access switch to which a node is connected.
        """
        topology = self.topology()
        if v not in topology.nodes():
            return None
        stack_name, stack_props = fnss.get_stack(topology, v)
        if stack_name == "receiver":
            if "sw" in stack_props:
                return stack_props["sw"]
        # If the node is not a receiver, alert the user
        logger.warning("Node %s is not a receiver or does not have attribute 'sw'!", v)

    def get_asn(self, v):
        """Return the ASN of a node
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        v : Node

        Returns
        -------
        asn : int
            The ASN of the node
        """
        return self.model.as_num[v]

    def get_ctrln(self, v):
        """Return the controller to which a node is connected.
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        v : Node

        Returns
        -------
        ctrln : any hashable type
            The controller to which the node is connected
        """
        return self.model.ctrl_num[v]

    def get_related_content(self, p_content, cache_node, server_node, **kwargs):
        """
        Parameters
        ----------
        p_content:
            The primary content name
        cache_node:
            The cache node name
        server_node:
            The server node name
        kwargs:
            k: the number of related content
            method: the method to choose related content, random, optimal or recommended
            index: the index of the workload (only use in optimal mode)
            tm: timestamp
        Returns:
            ret: The related content list: [(c1,v1),(c2,v2),...,(ck,vk)]
        """
        ret = []
        k = kwargs.get("k", 50)
        contents = self.model.source_node.get(server_node, [])
        if len(contents) == 0:
            logger.warning("The server node %s has no content!", server_node)
        # todo: five method, 1. random 2. optimal 3. popularity 4. recommend, 5. group
        # * ---- 1. random ----
        if kwargs.get("method", "random") == "random":
            # choose k content randomly from contents without p_content, set value to 1
            if len(contents) <= k:
                ret = [(c, 1) for c in contents if c != p_content]
            else:
                ret = [(c, 1) for c in random.sample(contents, k) if c != p_content]
        # * ---- 2. optimal ----
        elif kwargs.get("method", "random") == "optimal":
            # Select the content of the last k requests in the workload without p_content, set value to 1
            if len(contents) <= k:
                ret = [(c, 1) for c in contents if c != p_content]
            else:
                if "index" == -1:
                    raise ValueError("The index is not provided! in optimal mode")
                index = kwargs.get("index")
                if not hasattr(self.workload(), "reqs_df"):
                    raise ValueError("Optimal mode should use a REAL workload!")
                reqs_df = self.workload().reqs_df
                # return the following k content in the reqs_df from index which has the same value "cache_node" in the column "city"
                # and the value of the column "uri" is not equal to p_content
                df = reqs_df.iloc[index + 1:]
                df = df[df["city"].eq(cache_node) & ~df["uri"].eq(p_content)]
                df = df.head(k)

                # print("cache_node:{}, index:{}".format(cache_node, index))
                # print(df)
                ret = [(c, 1) for c in df["uri"]]
        # * ---- 3. popularity ----
        # send additional k content to the cache node according to the popularity of the content
        # always send the content with the highest popularity in global scope
        elif kwargs.get("method", "random") == "popularity":
            ret = self.get_global_content_pop(k)

        # * ---- 4. recommend ----
        elif kwargs.get("method", "random") == "recommend":
            tm = kwargs.get("tm", -1)
            if tm == -1:
                raise ValueError("The time is not provided! in recommend mode")
            ret = self.get_recommend_content(k, cache_node, p_content, tm)

        # * ---- 5. group ----
        elif kwargs.get("method", "random") == "group":
            if "index" == -1:
                raise ValueError("The index is not provided! in optimal mode")
            index = kwargs.get("index")
            if not hasattr(self.workload(), "reqs_df"):
                raise ValueError("Optimal mode should use a REAL workload!")
            reqs_df = self.workload().reqs_df
            # return the following k content in the reqs_df from index which has the same value "cache_node" in the column "city"
            # and the value of the column "uri" is not equal to p_content
            df = reqs_df.iloc[index + 1:index + 1 + k]
            ret = [(c, 1) for c in df["uri"]]

        else:
            raise ValueError("The method is not supported!" % kwargs.get("method", "random"))
        return ret

    def get_content_freq(self, k):
        """Return the frequency of content requests

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        freq : int
            The frequency of content requests
        """
        timestamp_l = self.model.content_freq[k]
        if len(timestamp_l) == 0 or len(timestamp_l) == 1:
            return 0
        return len(timestamp_l) / (timestamp_l[-1] - timestamp_l[0])

    def get_dst_freq(self, v):
        """Return the frequency of choose node v as destination node

        Parameters
        ----------
        v : int
            The destination node

        Returns
        -------
        freq : int
            The frequency of choose node v as destination node
        """
        return self.model.dst_node_freq_tracker.get_instantaneous_frequency(v, Sim_T.get_sim_time())

    def get_content_pop(self, k):
        """Return the popularity of content

        Parameters
        ----------
        k : any hashable type
            The content identifier

        Returns
        -------
        pop : float
            The popularity of content requests
        """
        timestamp_l = self.model.content_freq[k]
        if len(timestamp_l) == 0 or len(timestamp_l) == 1:
            return 0
        t_end = timestamp_l[-1]
        t_start = t_end - NetworkView.POP_RANGE if t_end > NetworkView.POP_RANGE else 0
        # find t_start in timestamp_l use bisect
        c_freq = len(timestamp_l) - bisect.bisect_right(timestamp_l, t_start)
        total_freq = len(self.model.req_freq) - bisect.bisect_right(self.model.req_freq, t_start)
        pop = c_freq / total_freq if total_freq != 0 else 0
        # normalizing pop
        if total_freq > 1000:
            self.model.content_pop_max = max(self.model.content_pop_max, pop)
        if self.model.content_pop_max != 0:
            pop /= self.model.content_pop_max
        # print("t_start:{}, t_end:{}, total_req_num:{}, pop:{}, pop_max: {}".format(t_start, t_end, total_freq, pop, self.model.content_pop_max))
        return min(pop, 1.0)

    def get_global_content_pop(self, k):
        """Return the global popularity of content

        Parameters
        ----------
        k : int
            The number of content
        Returns
        -------
        pop_l : list
            The popularity of content in [(c1, p1),(c2, p2),...,(ck, pk),...]
        """

        return self.model.content_pop[:k]

    def get_recommend_content(self, k, cache_node, p_content, time):
        # open pred_file and get the recommend content, each line in pred_file is a recommend content list
        # Each line's format is [(content, value), (content, value), ...]
        rec_val_dict = self.workload().rec_val_dict
        time_uri_dict = self.workload().group_uri_dict  # key is timestamp group, value is a list of uri
        uri2time_dict = self.workload().uri2time_dict  # key is uri, value is timestamp
        if rec_val_dict is None or time_uri_dict is None:
            raise ValueError("Recommend dataframe is None! check config file!")

        # get random k recommend content in line cache_node without p_content
        timestamps = uri2time_dict[p_content]
        # filter timestamp before time arg
        timestamps = [t for t in timestamps if self.workload().start_time < t < time]
        uri_t_filter = set()
        for it, t in enumerate(timestamps):
            if it > 20:
                break
            # range is 10 seconds
            for i in range(t, int(min(t + 10, time))):
                if i in time_uri_dict:
                    uri_t_filter.update(time_uri_dict[i])
        # ret is half of rec_val_dict[cache_node] and half of uri_t_filter
        # ret = ret[:k // 2] + [(c, 1) for c in uri_t_filter][:k // 2]
        ret = [(c, 1) for c in uri_t_filter]
        if len(ret) < k:
            ret += rec_val_dict[cache_node][:k - len(ret)]
        # ret = [c for c in ret if c[0] in uri_t_filter]
        # random.shuffle(ret)
        return ret[:k]

    def get_content_ttl(self, v, k):
        """Return the TTL of content

        Parameters
        ----------
        v : any hashable type
            The node identifier

        k : any hashable type
            The content identifier

        Returns
        -------
        ttl : int
            The TTL of content
        """
        if v not in self.model.cache:
            return 0
        if not hasattr(self.model.cache[v], "content_ttl"):
            return 0
        return self.model.cache[v].content_ttl[k]

    def get_default_ttl(self):
        """Return the default TTL of content

        Returns
        -------
        ttl : int
            The default TTL of content
        """
        return self.model.t0

    def get_switches_in_ctrl(self, asn, ctrln):
        """Return the switches in the given controller
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        asn : int
            The ASN
        ctrln : int
            The controller number

        Returns
        -------
        sw : set
            The switches set in the given controller
        """
        return self.model.switches_in_ctrl[asn][ctrln]

    def get_bgns_in_as(self, asn):
        """Return the BGP routers in the given ASN
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        asn : int
            The ASN

        Returns
        -------
        bgn : any hashable type
            The BGP routers set in the given ASN
        """
        return self.model.bgn_nodes[asn]

    def get_bgn_conhash(self, content: str):
        """Return the conhash of a BGP router
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        content: 
            The content name
        Returns
        -------
            The bgn which is responsible for the content in inter-domain
        """
        return self.model.conhash.get_node(content)

    def get_mcf(self, bgn):
        """Return the MCF of a BGP router
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        bgn : node
            BGP router

        Returns
        -------
        mcf : dict
            The MCF of the BGP router
        """
        return self.model.MCFS[bgn]

    def get_dst_from_controller(self, asn, ctrln, content):
        """Return the resolve node from the controller
        ---- Only used in SEANet topology. ----
        Parameters
        ----------
        asn : int
            The ASN
        ctrln : int
            The controller number
        content : str
        Returns
        -------
        node : node of topology
            The identifier of the destination node
        """
        sdn_controller = self.model.sdncontrollers[asn][ctrln]
        if content in sdn_controller:
            return sdn_controller[content]

    def get_dhts(self):
        """Return the DHTs
        ---- Only used in MDHT topology. ----
        Returns
        -------
        dhts : dict
            The DHTs
        """
        return self.model.dhts

    def dn2tn(self, dht_str: str) -> int:
        """
        ---- Only used in MDHT topology. ----
        return topology node id from dht node str, like: 'l1@3@1@174'
        """
        return self.model.dn2tn[dht_str]

    def shortest_path(self, s, t):
        """Return the shortest path from *s* to *t*

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node

        Returns
        -------
        shortest_path : list
            List of nodes of the shortest path (origin and destination
            included)
        """
        return self.model.shortest_path[s][t]

    def all_pairs_shortest_paths(self):
        """Return all pairs shortest paths

        Return
        ------
        all_pairs_shortest_paths : dict of lists
            Shortest paths between all pairs
        """
        return self.model.shortest_path

    def cluster(self, v):
        """Return cluster to which a node belongs, if any

        Parameters
        ----------
        v : any hashable type
            Node

        Returns
        -------
        cluster : int
            Cluster to which the node belongs, None if the topology is not
            clustered or the node does not belong to any cluster
        """
        if "cluster" in self.model.topology.node[v]:
            return self.model.topology.node[v]["cluster"]
        else:
            return None

    def link_type(self, u, v):
        """Return the type of link *(u, v)*.

        Type can be either *internal* or *external*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        link_type : str
            The link type
        """
        return self.model.link_type[(u, v)]

    def link_delay(self, u, v):
        """Return the delay of link *(u, v)*.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        delay : float
            The link delay
        """
        return self.model.link_delay[(u, v)]

    def link_capacity(self, u, v):
        """Return the capacity of link *(u, v)*.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node

        Returns
        -------
        capacity : float
            The link capacity
        """
        if (u, v) not in self.model.link_capacity:
            return 0
        return self.model.link_capacity[(u, v)]

    def link_capacity_avail(self, u, v, ts):
        """Return the available capacity of link *(u, v)*.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        ts : float
            The time at which the capacity is queried
        Returns
        -------
        capacity : float
            The link capacity
        """
        total_cap = max(self.link_capacity(u, v), self.link_capacity(v, u))
        if total_cap == 0:
            logger.warning("Link (%s, %s) has zero capacity!", u, v)
        used_capacity = 0
        if (u, v) in self.model.link_capacity_used:
            used_ts_cp_l = self.model.link_capacity_used[(u, v)]  # [(timestamp, capacity), ...]
            # use bisect to find the index of ts in self.model.link_capacity_used[(u, v)]
            index = bisect.bisect_right([ts for ts, _ in used_ts_cp_l], ts)
            # and sum the capacity before ts
            used_capacity = sum([cp for _, cp in used_ts_cp_l[:index]])
        elif (v, u) in self.model.link_capacity_used:
            used_ts_cp_l = self.model.link_capacity_used[(v, u)]
            index = bisect.bisect_right([ts for ts, _ in used_ts_cp_l], ts)
            used_capacity = sum([cp for _, cp in used_ts_cp_l[:index]])
        # print("test---------------used_capacity:{}".format(used_capacity))
        return total_cap - used_capacity

    def link_loss_rate(self, u, v):
        return 0

    def topology(self):
        """Return the network topology

        Returns
        -------
        topology : fnss.Topology
            The topology object

        Notes
        -----
        The topology object returned by this method must not be modified by the
        caller. This object can only be modified through the NetworkController.
        Changes to this object will lead to inconsistent network state.
        """
        return self.model.topology

    def workload(self):
        """Return the model workload

        Returns
        -------
        workload :
            The workload object

        """
        return self.model.workload

    def cache_nodes(self, size=False):
        """Returns a list of nodes with caching capability

        Parameters
        ----------
        size: bool, opt
            If *True* return dict mapping nodes with size

        Returns
        -------
        cache_nodes : list or dict
            If size parameter is False or not specified, it is a list of nodes
            with caches. Otherwise it is a dict mapping nodes with a cache
            and their size.
        """
        return (
            {v: c.maxlen for v, c in self.model.cache.items()}
            if size
            else list(self.model.cache.keys())
        )

    def has_cache(self, node):
        """Check if a node has a content cache.

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        has_cache : bool,
            *True* if the node has a cache, *False* otherwise
        """
        return node in self.model.cache

    def cache_lookup(self, node, content):
        """Check if the cache of a node has a content object, without changing
        the internal state of the cache.

        This method is meant to be used by data collectors to calculate
        metrics. It should not be used by strategies to look up for contents
        during the simulation. Instead they should use
        `NetworkController.get_content`

        Parameters
        ----------
        node : any hashable type
            The node identifier
        content : any hashable type
            The content identifier

        Returns
        -------
        has_content : bool
            *True* if the cache of the node has the content, *False* otherwise.
            If the node does not have a cache, return *None*
        """
        if node in self.model.cache:
            return self.model.cache[node].has(content)

    def local_cache_lookup(self, node, content):
        """Check if the local cache of a node has a content object, without
        changing the internal state of the cache.

        The local cache is an area of the cache of a node reserved for
        uncoordinated caching. This is currently used only by hybrid
        hash-routing strategies.

        This method is meant to be used by data collectors to calculate
        metrics. It should not be used by strategies to look up for contents
        during the simulation. Instead they should use
        `NetworkController.get_content_local_cache`.

        Parameters
        ----------
        node : any hashable type
            The node identifier
        content : any hashable type
            The content identifier

        Returns
        -------
        has_content : bool
            *True* if the cache of the node has the content, *False* otherwise.
            If the node does not have a cache, return *None*
        """
        if node in self.model.local_cache:
            return self.model.local_cache[node].has(content)
        else:
            return False

    def get_available_cache_size(self, node):
        """
        Returns the available cache size of a node.
        """
        if node in self.model.cache:
            # if ttl cache, purge expired items first
            if hasattr(self.model.cache[node], "expiry"):
                self.model.cache[node].purge()
            return self.model.cache[node].maxlen - len(self.model.cache[node])

    def get_all_switches_free_space_ratio(self):
        """
        Returns the ratio of available cache space for each switch.
        """
        switches = self.cache_nodes()
        total_used_space = sum([len(self.model.cache[s]) for s in switches])
        total_space = sum([self.model.cache[s].maxlen for s in switches])
        return (total_space - total_used_space) / total_space

    def cache_dump(self, node):
        """Returns the dump of the content of a cache in a specific node

        Parameters
        ----------
        node : any hashable type
            The node identifier

        Returns
        -------
        dump : list
            List of contents currently in the cache
        """
        if node in self.model.cache:
            return self.model.cache[node].dump()


class NetworkModel:
    """Models the internal state of the network.

    This object should never be edited by strategies directly, but only through
    calls to the network controller.
    """

    def __init__(self, topology, cache_policy, workload, shortest_path=None):
        """Constructor

        Parameters
        ----------
        topology : fnss.Topology
            The topology object
        cache_policy : dict or Tree
            cache policy descriptor. It has the name attribute which identify
            the cache policy name and keyworded arguments specific to the
            policy
        shortest_path : dict of dict, optional
            The all-pair shortest paths of the network
        """
        self.t0 = 10  # for ttl cache
        # Filter inputs
        if not isinstance(topology, fnss.Topology):
            raise ValueError(
                "The topology argument must be an instance of "
                "fnss.Topology or any of its subclasses."
            )

        # Shortest paths of the network
        logger.info("Calculate all pairs of dijkstra_path...")
        self.shortest_path = (
            dict(shortest_path)
            if shortest_path is not None
            else symmetrify_paths(dict(nx.all_pairs_dijkstra_path(topology)))
        )
        logger.info("Calculate all pairs of dijkstra_path done!")

        # Network topology
        self.topology = topology
        # Workload
        self.workload = workload
        # Dictionary mapping each content object to its source
        # dict of location of contents keyed by content ID
        self.content_source = defaultdict(list)
        # Dictionary mapping the reverse, i.e. nodes to set of contents stored
        self.source_node = {}
        # Dictionary mapping each content request frequency {c: [t1, t2, ...]}
        self.content_freq = defaultdict(list)
        # capture the timestamp of each request
        self.req_freq = []
        # list of content objects and their popularity [(c1, p1), (c2, p2), ...], read from file
        self.content_pop = []
        self.content_pop_max = 0
        # frequency of choose node v as destination node {v: [t1, t2, ...]}
        self.dst_node_freq_tracker = TimeWindowFrequencyTracker()

        # Dictionary of link types (internal/external)
        self.link_type = nx.get_edge_attributes(topology, "type")
        self.link_delay = fnss.get_delays(topology)
        # Dictionary of link capacities
        self.link_capacity = fnss.get_capacities(topology)
        # 创建一个字典，key是边，value是边的每个时刻的消耗带宽，正值表示带宽消耗，负值表示带宽释放，按照时刻排序
        self.link_capacity_used = defaultdict(SortedList)
        # Instead of this manual assignment, I could have converted the
        # topology to directed before extracting type and link delay but that
        # requires a deep copy of the topology that can take long time if
        # many content source mappings are included in the topology
        if not topology.is_directed():
            for (u, v), link_type in list(self.link_type.items()):
                self.link_type[(v, u)] = link_type
            for (u, v), delay in list(self.link_delay.items()):
                self.link_delay[(v, u)] = delay

        cache_size = {}
        for node in tqdm(topology.nodes(), desc="Mapping content sources to nodes: "):
            stack_name, stack_props = fnss.get_stack(topology, node)
            if stack_name == "router" or stack_name == "switch":
                if "cache_size" in stack_props:
                    cache_size[node] = stack_props["cache_size"]
            elif stack_name == "source":
                contents = stack_props.get("contents", [])
                self.source_node[node] = contents
                for content in contents:
                    self.content_source[content].append(node)
        if any(c < 1 for c in cache_size.values()):
            logger.warning(
                "Some content caches have size equal to 0. "
                "I am setting them to 1 and run the experiment anyway"
            )
            for node in cache_size:
                if cache_size[node] < 1:
                    cache_size[node] = 1

        policy_name = cache_policy["name"]
        policy_args = {k: v for k, v in cache_policy.items() if k != "name"}
        # The actual cache objects storing the content
        self.cache = {}
        for node in cache_size:
            nc = CACHE_POLICY[policy_name](cache_size[node], **policy_args)
            if "timeout" in policy_args and policy_args["timeout"]:
                self.t0 = policy_args["t0"]
                self.cache[node] = ttl_cache(nc, Sim_T.get_sim_time, t0=self.t0)
            else:
                self.cache[node] = nc

        # if REAL_Workload, get content popularity from workload
        if hasattr(workload, "content_popularity"):
            self.content_pop = workload.content_popularity

        # This is for a local un-coordinated cache (currently used only by
        # Hashrouting with edge cache)
        self.local_cache = {}

        # Keep track of nodes and links removed to simulate failures
        self.removed_nodes = {}
        # This keeps track of neighbors of a removed node at the time of removal.
        # It is needed to ensure that when the node is restored only links that
        # were removed as part of the node removal are restored and to prevent
        # restoring nodes that were removed manually before removing the node.
        self.disconnected_neighbors = {}
        self.removed_links = {}
        self.removed_sources = {}
        self.removed_caches = {}
        self.removed_local_caches = {}


class NetworkController:
    """Network controller

    This class is in charge of executing operations on the network model on
    behalf of a strategy implementation. It is also in charge of notifying
    data collectors of relevant events.
    """

    def __init__(self, model):
        """Constructor

        Parameters
        ----------
        model : NetworkModel
            Instance of the network model
        """
        self.session = None
        self.model = model
        self.collector = None
        self.counter = 0  # counter for the number of server hits

    def attach_collector(self, collector):
        """Attach a data collector to which all events will be reported.

        Parameters
        ----------
        collector : DataCollector
            The data collector
        """
        self.collector = collector

    def detach_collector(self):
        """Detach the data collector."""
        self.collector = None

    def start_session(self, timestamp, receiver, content, log, **kwargs):
        """Instruct the controller to start a new session (i.e. the retrieval
        of a content).

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        receiver : any hashable type
            The receiver node requesting a content
        content : any hashable type
            The content identifier requested by the receiver
        log : bool
            *True* if this session needs to be reported to the collector,
            *False* otherwise
        kwargs: dict
            Additional keyworded arguments
        """
        self.session = dict(
            timestamp=timestamp, receiver=receiver, content=content, log=log
        )
        if self.collector is not None and self.session["log"]:
            self.collector.start_session(timestamp, receiver, content, **kwargs)

    def forward_request_path(self, s, t, path=None, main_path=True):
        """Forward a request from node *s* to node *t* over the provided path.

        Parameters
        ----------
        s : any hashable type
            Origin node
        t : any hashable type
            Destination node
        path : list, optional
            The path to use. If not provided, shortest path is used
        main_path : bool, optional
            If *True*, indicates that link path is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        if path is None:
            path = self.model.shortest_path[s][t]
        for u, v in path_links(path):
            self.forward_request_hop(u, v, main_path)

    def forward_content_path(self, u, v, path=None, main_path=True):
        """Forward a content from node *s* to node *t* over the provided path.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        path : list, optional
            The path to use. If not provided, shortest path is used
        main_path : bool, optional
            If *True*, indicates that this path is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        if path is None:
            path = self.model.shortest_path[u][v]
        for u, v in path_links(path):
            self.forward_content_hop(u, v, main_path)

    def forward_request_hop(self, u, v, main_path=True):
        """Forward a request over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that link link is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        if self.collector is not None and self.session["log"]:
            self.collector.request_hop(u, v, main_path)

    def forward_content_hop(self, u, v, main_path=True):
        """Forward a content over link  u -> v.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that this link is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        if self.collector is not None and self.session["log"]:
            self.collector.content_hop(u, v, main_path)

    def packet_in(self, content):
        """Instruct the controller to process a packet in event.

        Parameters
        ----------
        content : any hashable type
            The content identifier of the packet
        """
        if self.collector is not None and self.session["log"]:
            self.collector.packet_in(content)

    def resolve(self, content, area):
        """Instruct the controller to resolve a content identifier.

        Parameters
        ----------
        content : any hashable type
            The content identifier to resolve
        area : any hashable type
            The area in which the content is being resolved, support: {"ctrl", "ibgn", "ebgn"}
        """
        if self.collector is not None and self.session["log"]:
            self.collector.resolve(content, area)

    def cal_free_space_ratio(self, timestamp):
        """Calculate the free space ratio of all ICN node.

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        """
        if self.collector is not None and self.session["log"]:
            self.collector.seq_hit_ratio(timestamp)

    def put_content(self, node, **kwargs):
        """Store content in the specified node.

        The node must have a cache stack and the actual insertion of the
        content is executed according to the caching policy. If the caching
        policy has a selective insertion policy, then content may not be
        inserted.

        Parameters
        ----------
        node : any hashable type
            The node where the content is inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        content = kwargs.get("content", self.session["content"])
        if node in self.model.cache:
            return self.model.cache[node].put(content, **kwargs)

    def get_content(self, node):
        """Get a content from a server or a cache.

        Parameters
        ----------
        node : any hashable type
            The node where the content is retrieved

        Returns
        -------
        content : bool
            True if the content is available, False otherwise
        """
        self.model.content_freq[self.session["content"]].append(Sim_T.get_sim_time())
        self.model.req_freq.append(Sim_T.get_sim_time())
        # calculate the hit ratio with the first 10 hits and then every 100 hits
        # self.collector.seq_hit_ratio(Sim_T.get_sim_time())
        self.counter += 1
        if self.counter < 200 or self.counter % 1000 == 0:
            self.collector.seq_hit_ratio(Sim_T.get_sim_time())
        if node in self.model.cache:
            cache_hit = self.model.cache[node].get(self.session["content"])
            if cache_hit:
                if self.session["log"]:
                    self.collector.cache_hit(node)
            else:
                if self.session["log"]:
                    self.collector.cache_miss(node)
            # self.collector.seq_hit_ratio(Sim_T.get_sim_time())
            return cache_hit
        name, props = fnss.get_stack(self.model.topology, node)
        if name == "source" and self.session["content"] in props["contents"]:
            if self.collector is not None and self.session["log"]:
                self.collector.server_hit(node)
            return True
        else:
            return False

    def remove_content(self, node):
        """Remove the content being handled from the cache

        Parameters
        ----------
        node : any hashable type
            The node where the cached content is removed

        Returns
        -------
        removed : bool
            *True* if the entry was in the cache, *False* if it was not.
        """
        if node in self.model.cache:
            return self.model.cache[node].remove(self.session["content"])

    def end_session(self, success=True):
        """Close a session

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        if self.collector is not None and self.session["log"]:
            self.collector.end_session(success)
        self.session = None

    def rewire_link(self, u, v, up, vp, recompute_paths=True):
        """Rewire an existing link to new endpoints

        This method can be used to model mobility patters, e.g., changing
        attachment points of sources and/or receivers.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact as a result of link rewiring, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        Parameters
        ----------
        u, v : any hashable type
            Endpoints of link before rewiring
        up, vp : any hashable type
            Endpoints of link after rewiring
        """
        link = self.model.topology.adj[u][v]
        self.model.topology.remove_edge(u, v)
        self.model.topology.add_edge(up, vp, **link)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def remove_link(self, u, v, recompute_paths=True):
        """Remove a link from the topology and update the network model.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact as a result of link removal, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        Also, note that, for these changes to be effective, the strategy must
        use fresh data provided by the network view and not storing local copies
        of network state because they won't be updated by this method.

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.removed_links[(u, v)] = self.model.topology.adj[u][v]
        self.model.topology.remove_edge(u, v)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def restore_link(self, u, v, recompute_paths=True):
        """Restore a previously-removed link and update the network model

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.topology.add_edge(u, v, **self.model.removed_links.pop((u, v)))
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def remove_node(self, v, recompute_paths=True):
        """Remove a node from the topology and update the network model.

        Note well. With great power comes great responsibility. Be careful when
        using this method. In fact, as a result of node removal, network
        partitions and other corner cases might occur. Ensure that the
        implementation of strategies using this method deal with all potential
        corner cases appropriately.

        It should be noted that when this method is called, all links connected
        to the node to be removed are removed as well. These links are however
        restored when the node is restored. However, if a link attached to this
        node was previously removed using the remove_link method, restoring the
        node won't restore that link as well. It will need to be restored with a
        call to restore_link.

        This method is normally quite safe when applied to remove cache nodes or
        routers if this does not cause partitions. If used to remove content
        sources or receiver, special attention is required. In particular, if
        a source is removed, the content items stored by that source will no
        longer be available if not cached elsewhere.

        Also, note that, for these changes to be effective, the strategy must
        use fresh data provided by the network view and not storing local copies
        of network state because they won't be updated by this method.

        Parameters
        ----------
        v : any hashable type
            Node to remove
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.removed_nodes[v] = self.model.topology.node[v]
        # First need to remove all links the removed node as endpoint
        neighbors = self.model.topology.adj[v]
        self.model.disconnected_neighbors[v] = set(neighbors.keys())
        for u in self.model.disconnected_neighbors[v]:
            self.remove_link(v, u, recompute_paths=False)
        self.model.topology.remove_node(v)
        if v in self.model.cache:
            self.model.removed_caches[v] = self.model.cache.pop(v)
        if v in self.model.local_cache:
            self.model.removed_local_caches[v] = self.model.local_cache.pop(v)
        if v in self.model.source_node:
            self.model.removed_sources[v] = self.model.source_node.pop(v)
            for content in self.model.removed_sources[v]:
                self.model.content_source.pop(content)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def restore_node(self, v, recompute_paths=True):
        """Restore a previously-removed node and update the network model.

        Parameters
        ----------
        v : any hashable type
            Node to restore
        recompute_paths: bool, optional
            If True, recompute all shortest paths
        """
        self.model.topology.add_node(v, **self.model.removed_nodes.pop(v))
        for u in self.model.disconnected_neighbors[v]:
            if (v, u) in self.model.removed_links:
                self.restore_link(v, u, recompute_paths=False)
        self.model.disconnected_neighbors.pop(v)
        if v in self.model.removed_caches:
            self.model.cache[v] = self.model.removed_caches.pop(v)
        if v in self.model.removed_local_caches:
            self.model.local_cache[v] = self.model.removed_local_caches.pop(v)
        if v in self.model.removed_sources:
            self.model.source_node[v] = self.model.removed_sources.pop(v)
            for content in self.model.source_node[v]:
                self.model.content_source[content].append(v)
        if recompute_paths:
            shortest_path = dict(nx.all_pairs_dijkstra_path(self.model.topology))
            self.model.shortest_path = symmetrify_paths(shortest_path)

    def reserve_local_cache(self, ratio=0.1):
        """Reserve a fraction of cache as local.

        This method reserves a fixed fraction of the cache of each caching node
        to act as local uncoordinated cache. Methods `get_content` and
        `put_content` will only operated to the coordinated cache. The reserved
        local cache can be accessed with methods `get_content_local_cache` and
        `put_content_local_cache`.

        This function is currently used only by hybrid hash-routing strategies.

        Parameters
        ----------
        ratio : float
            The ratio of cache space to be reserved as local cache.
        """
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")
        for v, c in list(self.model.cache.items()):
            maxlen = iround(c.maxlen * (1 - ratio))
            if maxlen > 0:
                self.model.cache[v] = type(c)(maxlen)
            else:
                # If the coordinated cache size is zero, then remove cache
                # from that location
                if v in self.model.cache:
                    self.model.cache.pop(v)
            local_maxlen = iround(c.maxlen * (ratio))
            if local_maxlen > 0:
                self.model.local_cache[v] = type(c)(local_maxlen)

    def get_content_local_cache(self, node):
        """Get content from local cache of node (if any)

        Get content from a local cache of a node. Local cache must be
        initialized with the `reserve_local_cache` method.

        Parameters
        ----------
        node : any hashable type
            The node to query
        """
        if node not in self.model.local_cache:
            return False
        cache_hit = self.model.local_cache[node].get(self.session["content"])
        if cache_hit:
            if self.session["log"]:
                self.collector.cache_hit(node)
        else:
            if self.session["log"]:
                self.collector.cache_miss(node)
        return cache_hit

    def put_content_local_cache(self, node):
        """Put content into local cache of node (if any)

        Put content into a local cache of a node. Local cache must be
        initialized with the `reserve_local_cache` method.

        Parameters
        ----------
        node : any hashable type
            The node to query
        """
        if node in self.model.local_cache:
            return self.model.local_cache[node].put(self.session["content"])

    def update_dst_freq(self, v, timestamp):
        """Add timestamp to frequency tracker.

        Parameters
        ----------
        v : any hashable type
            The node to query
        timestamp : int
            The timestamp of the event
        """
        self.model.dst_node_freq_tracker.add_timestamp(v, timestamp)

    def update_link_capacity_used(self, u, v, timestamp, capacity):
        """Update the link capacity used.

        Parameters
        ----------
        u, v : any hashable type
            endpoints of link
        timestamp : int
            The timestamp of the event
        capacity : float
            The capacity to be updated
        """
        self.model.link_capacity_used[(u, v)].add((timestamp, capacity))


# A ICN-SEANet NETWORK MODEL CLASS extent NetworkModel
class SEANRSModel(NetworkModel):
    """
    ICN-SEANet Name Resolution System Network Model

    This class extends the NetworkModel class to add some new features in SEANet INNRS
    """

    def __init__(self, topology, cache_policy, workload, shortest_path=None):
        logger.info("Initializing SEANRS Network Model...")
        super().__init__(topology, cache_policy, workload, shortest_path)
        # controllers of global specific name space
        # sdncontrollers[as_id][ctrl_id] = {content: node}
        self.sdncontrollers = defaultdict(lambda: defaultdict(dict))
        # MCFs(Marked Cuckoo Filters) of global specific name space
        # MCFs[BGN_node] = MCF()
        self.MCFS = {}
        # consistent hash ring of bgns
        self.conhash = HashRing(hash_fn=mmh3.hash)
        # access-switch of receiver
        # todo: Temporary set one receiver only has one access-switch, but maybe not.
        self.access_switches = {}
        # access-switches of controller, switches_in_ctrl[as_id][ctrl_id] = set()
        self.switches_in_ctrl = defaultdict(lambda: defaultdict(set))
        # as number of each node, as_num[node] = as_id
        self.as_num = {}
        # controller number of switch, ctrl_num[node] = ctrl_id
        self.ctrl_num = {}
        # BGN node of each as, bgn_nodes[as_id] = BGN_node
        self.bgn_nodes = defaultdict(set)
        # logger.info("Set function of DINNRS controller and BGN ...")
        for node in tqdm(topology.nodes(), desc="[Network] Set controllers and BGNs: "):
            stack_name, stack_props = fnss.get_stack(topology, node)
            self.as_num[node] = stack_props['asn']
            if stack_name == "receiver":
                self.access_switches[node] = topology.node[stack_props["sw"]]
            elif stack_name == "bgn":
                # todo: perhaps not all bgn nodes are members of the intermediary domain
                self.conhash.add_node(node, conf=stack_props)
                self.bgn_nodes[stack_props['asn']].add(node)
                # ! set MDCF of each BGN node, capacity default to 10000
                self.MCFS[node] = ScalableCuckooFilter(100000, 10 ** (-6), class_type=MarkedCuckooFilter,
                                                       bit_tag_len=64)
            elif stack_name == "switch":
                self.ctrl_num[node] = stack_props["ctrl"]
                self.switches_in_ctrl[stack_props['asn']][stack_props["ctrl"]].add(node)
            elif stack_name == "source":
                self.ctrl_num[node] = stack_props["ctrl"]
                # ! register content to sdn controller
                for content in stack_props.get("contents", []):
                    self.sdncontrollers[stack_props['asn']][stack_props['ctrl']][content] = node
            elif stack_name == "router":
                # todo: router is not considered in this version
                pass
            else:
                logger.warning("[SEANRS] Unknown stack name: %s", stack_name)
        # ! register content to MDCF of BGN node
        # logger.info("Register content to MDCF of BGN node ...")
        for node in tqdm(self.source_node.keys(), desc="[Network] Register content to MDCF: "):
            asn = self.as_num[node]
            for content in self.source_node[node]:
                # * ----- intra-domain -----
                for bgn_node in self.bgn_nodes[asn]:
                    mcf = self.MCFS[bgn_node]
                    if node in self.ctrl_num:
                        mask = mcf.encode_mask("bit", self.ctrl_num[node])
                        mcf.insert(str(content), mask=mask)
                    else:
                        logger.warning("[SEANRS] No controller number of source node: %s", node)
                # * ----- inter-domain ------
                # consistent Hashing content to a bgn
                ibgn = self.conhash.get_node(str(content))
                # register to inter-domain bgn 
                mcf = self.MCFS[ibgn]
                mask = mcf.encode_mask("int", asn)
                mcf.insert(str(content), mask=mask)

        logger.info("SEANet DINNRS model init success!")


class MDHTModel(NetworkModel):
    """
    Paper [1] MDHT: A Hierarchical Name Resolution Service for Information-centric Networks
    MDHT Name Resolution System Network Model
    3 levels of MDHT: BGN O(log(n)), AS O(1), and ctrl O(1)
    """

    def __init__(self, topology, cache_policy, workload, shortest_path=None, k=(10, 10, 15)):
        """
        Parameters:
            k: 3 levels, 2 ** k is the number of buckets in each level of MDHT
        """
        super().__init__(topology, cache_policy, workload, shortest_path)
        self.dn2tn = {}  # dht node id to topology node id
        # as number of each node, as_num[node] = as_id
        self.as_num = {}
        # controller number of switch, ctrl_num[node] = ctrl_id
        self.ctrl_num = {}
        # create DHTs in 3 levels and add nodes to them
        self.dhts = defaultdict(dict)
        for node in topology.nodes():
            # get 3 levels dht ids of node
            dht_ids = [MDHTModel.get_hash_id(node, k[i]) for i in range(3)]
            # ! >>> level 1: DHTs of ctrl-domain
            stack_name, stack_props = fnss.get_stack(topology, node)
            self.as_num[node] = stack_props["asn"]
            as_id = str(stack_props["asn"])
            if stack_name == "receiver" or stack_name == "source":
                self.ctrl_num[node] = stack_props["ctrl"]
                continue
            if "ctrl" in stack_props:
                self.ctrl_num[node] = stack_props["ctrl"]
                ctrl_id = str(stack_props["ctrl"])
                if as_id + "@" + ctrl_id not in self.dhts[1]:
                    # todo: how to set DHT size?
                    self.dhts[1][as_id + "@" + ctrl_id] = DHT(k[0], dht_ids[0])
                    self.dn2tn["l1@" + as_id + "@" + ctrl_id + "@" + str(dht_ids[0])] = node
                # print("add node %s/%s to dht %s" % (node, dht_ids[0], as_id + "@" + ctrl_id))
                if self.dhts[1][as_id + "@" + ctrl_id].join(NodeDHT(dht_ids[0])):
                    self.dn2tn["l1@" + as_id + "@" + ctrl_id + "@" + str(dht_ids[0])] = node
            # ! >>> level 2: DHTs of each as
            if as_id not in self.dhts[2]:
                self.dhts[2][as_id] = DHT(k[1], dht_ids[1])
                self.dn2tn["l2@" + as_id + "@" + str(dht_ids[1])] = node
            if self.dhts[2][as_id].join(NodeDHT(dht_ids[1])):
                self.dn2tn["l2@" + as_id + "@" + str(dht_ids[1])] = node
            # ! >>> level 3: DHTs of global
            if "G" not in self.dhts[3]:
                self.dhts[3]["G"] = DHT(k[2], dht_ids[2])
                self.dn2tn["l3@" + str(dht_ids[2])] = node
            if self.dhts[3]["G"].join(NodeDHT(dht_ids[2])):
                self.dn2tn["l3@" + str(dht_ids[2])] = node

        # update finger tables of DHTs
        for level in range(1, 4):
            for dht in self.dhts[level].values():
                dht.update_all_finger_tables()
                # print("number of nodes: {} in dht {}".format(dht.get_num_nodes(), dht))

        # register contents
        for node in self.source_node.keys():
            for content in self.source_node[node]:
                stack_name, stack_props = fnss.get_stack(topology, node)
                as_id = str(stack_props["asn"])
                if "ctrl" in stack_props:
                    ctrl_id = str(stack_props["ctrl"])
                    self.dhts[1][as_id + "@" + ctrl_id].store(str(content), node)
                self.dhts[2][as_id].store(str(content), node)
                self.dhts[3]["G"].store(str(content), node)

    @staticmethod
    def get_hash_id(node_id, k) -> int:
        """
        get hash id of node_id
        :param node_id: node id
        :param k: 2 ** k is the number of slots in the DHT
        :return: hash id
        """
        return int(hash(str(node_id)) % (2 ** k))
