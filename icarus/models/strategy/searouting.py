"""Implementations of SEANet strategies"""
import logging
import random

from icarus.execution.simulation_time import Sim_T
from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links
from .base import Strategy

__all__ = [
    "SEANRS",
    "SEACACHE",
    "SEALOC",
]

logger = logging.getLogger("SEANRS-strategy")


def normalize(link_status_map):
    """normalize values in map to [0, 1]"""
    max_val = max(link_status_map.values())
    min_val = min(link_status_map.values())
    for k, v in link_status_map.items():
        if max_val == min_val:
            link_status_map[k] = 0.5
        else:
            link_status_map[k] = (v - min_val) / (max_val - min_val)


@register_strategy("SEANRS")
class SEANRS(Strategy):
    """
    This strategy implements the SEANet routing strategy.
    """

    def __init__(self, view, controller, **kwargs):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        controller : NetworkController
            An instance of the network controller
        """
        super().__init__(view, controller)

    # resolve process in the control-domain
    def resolve_ctrl(self, sw):
        """Resolve a content request in the control domain

        Parameters
        ----------
        sw : Node
            The switch dealing with the request

        Returns
        -------
        node : Node
            The node(actually ip address) to route the request to
        """
        content = self.controller.session['content']
        # Check whether the content is in the cache
        if self.view.has_cache(sw) and self.controller.get_content(sw):
            self.controller.resolve(sw, "cache")
            # todo: this method will be changed to the content_location
            dst_n = self.view.content_source(content)
            return dst_n
        # Check whether the content is in the sdn controller
        self.controller.resolve(sw, "ctrl")
        self.controller.packet_in(content)
        dst_n = self.view.get_dst_from_controller(
            self.view.get_asn(sw), self.view.get_ctrln(sw), content)
        if dst_n:
            self.controller.get_content(dst_n)
            # Update the cache of the switch
            if self.view.has_cache(sw):
                self.controller.put_content(sw)
        return dst_n

    def resolve_bgn(self, bgn, type="ibgn"):
        """Resolve a content request in the bgp domain

        Parameters
        ----------
        bgn : Node
            The bgp node dealing with the request
        type: str
            Whether the request is from the control-domain or manage-domain
            control-domain -> ibgn, manage-domain -> ebgn

        Returns
        -------
        node_list : [Node]
            The valid access node list to route the request to
        bgn_next:
            The next bgp node to route the request to
        """

        def get_nearest_sw(_bgn, sw_set: set):
            nsw = None
            # if sw_set is empty, return None
            if not sw_set:
                return nsw
            min_path_len = float('inf')
            for sw in sw_set:
                # get the shortest path length
                path_len = len(path_links(self.view.shortest_path(_bgn, sw)))
                if path_len < min_path_len:
                    min_path_len = path_len
                    nsw = sw
            return nsw

        content = str(self.controller.session['content'])
        # check mcf(marked cuckoo filter) of bgn
        mcf = self.view.get_mcf(bgn)
        acc_sw = []  # acc_sw: [{1, 2, 3}, {4, 5, 6}, ...]
        n_sw_list = []  # nearest_sw: [1, 4, ...]
        bgn_next = None

        if type == "ibgn":
            self.controller.resolve(bgn, "ibgn")
            if mcf.contains(content):
                # if the content fingerprint is in the mcf, get the set mask from the mcf
                area, idx_l = mcf.decode_mask(mcf.get(content))
                if area == 'bit':
                    for idx in idx_l:
                        acc_sw.append(self.view.get_switches_in_ctrl(self.view.get_asn(bgn), idx))
                    # get the nearest switch for each valid controller domain
                    for sw_l in acc_sw:
                        if get_nearest_sw(bgn, sw_l):
                            n_sw_list.append(get_nearest_sw(bgn, sw_l))
        elif type == "ebgn":
            self.controller.resolve(bgn, "ebgn")
            if mcf.contains(content):
                area, idx_l = mcf.decode_mask(mcf.get(content))
                if area == 'bit':
                    for idx in idx_l:
                        acc_sw.append(self.view.get_switches_in_ctrl(self.view.get_asn(bgn), idx))
                    # get the nearest switch for each valid controller domain
                    for sw_l in acc_sw:
                        if get_nearest_sw(bgn, sw_l):
                            n_sw_list.append(get_nearest_sw(bgn, sw_l))
                else:
                    if idx_l[0] == 0:
                        logger.warning("False positive in the ebgn{}".format(bgn))
                        return []
                    # if resolve hit, get next manager-bgn from the mcf
                    bgn_next = random.choice(list(self.view.get_bgns_in_as(idx_l[0])))
                    mcf_next = self.view.get_mcf(bgn_next)
                    # route to bgn_next
                    self.controller.forward_request_path(bgn, bgn_next)
                    self.controller.resolve(bgn_next, "ibgn")
                    area_next, idx_l_next = mcf.decode_mask(mcf_next.get(content))
                    if area_next == 'bit':
                        for idx in idx_l_next:
                            # fixed: Not all control domains have access switches (topology problem)
                            sws = self.view.get_switches_in_ctrl(idx_l[0], idx)
                            if sws:
                                acc_sw.append(sws)
                        # get the nearest switch for each valid controller domain
                        for sw_l in acc_sw:
                            if get_nearest_sw(bgn_next, sw_l):
                                n_sw_list.append(get_nearest_sw(bgn_next, sw_l))
                    else:
                        logger.warning("False positive in the ebgn{}".format(bgn))
                        return [], []

        return n_sw_list, bgn_next

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log, **kwargs):
        """
        Process a content request event
        ===== Main process of SEANet routing strategy =====
        """
        # * >>> 1. route content to the access switch <<<
        acc_switch = self.view.get_access_switch(receiver)
        self.controller.start_session(time, receiver, content, log, **kwargs)
        # route the request from receiver to access switch
        self.controller.forward_request_path(receiver, acc_switch)

        # * >>> 2. If the content is resolved in the ctrl domain, get content and return <<<
        dst_node = self.resolve_ctrl(acc_switch)
        if dst_node:
            self.controller.forward_request_path(acc_switch, dst_node)
            # self.controller.get_content(dst_node)
            # forward content back to receiver
            self.controller.forward_content_path(dst_node, receiver)
            self.controller.end_session(success=True)
            return

        # * >>> 3. If the content is not resolved in the ctrl domain, get the content from the manage-domain <<<
        # get bgn of acc_switch
        bgns: set = self.view.get_bgns_in_as(self.view.get_asn(acc_switch))
        # todo: choose a bgn from bgns set, now choose the first one
        bgn = random.choice(list(bgns))
        # route the request to bgn
        self.controller.forward_request_path(acc_switch, bgn)
        # if the content is resolved in the manage-domain, get acc-switch
        nearest_sw_list, _ = self.resolve_bgn(bgn, type="ibgn")
        if len(nearest_sw_list) != 0:
            for sw in nearest_sw_list:
                self.controller.forward_request_path(bgn, sw)
                dst_node = self.resolve_ctrl(sw)
                if dst_node:
                    self.controller.forward_request_path(sw, dst_node)
                    break
                # if the content is not resolved in the foreign ctrl domain, false positive happens, route back to bgn
                self.controller.forward_request_path(sw, bgn)

        if dst_node:
            self.controller.forward_content_path(dst_node, receiver)
            self.controller.end_session(success=True)
            return
        # print("111 content:", content, "nearest_sw_list: ", nearest_sw_list)

        # * >>> 4. if the content is not resolved above, hash the content fingerprint to resolve in inter-domain bgn <<<
        inter_bgn = self.view.get_bgn_conhash(str(content))
        self.controller.forward_request_path(bgn, inter_bgn)
        nearest_sw_list, bgn_nxt = self.resolve_bgn(inter_bgn, type="ebgn")
        if len(nearest_sw_list) != 0:
            for sw in nearest_sw_list:
                if bgn_nxt:
                    self.controller.forward_request_path(bgn_nxt, sw)
                else:
                    self.controller.forward_request_path(inter_bgn, sw)
                dst_node = self.resolve_ctrl(sw)
                if dst_node:
                    self.controller.forward_request_path(sw, dst_node)
                    break
                # if the content is not resolved in the foreign ctrl domain, false positive happens, route back to bgn
                self.controller.forward_request_path(sw, inter_bgn)
        else:
            logger.warning('No valid access node found in the inter-domain bgn.'
                           'inter-bgn: %s, content: %s, source: %s' % (
                               inter_bgn, content, self.view.content_source(content)))
        if dst_node:
            self.controller.forward_content_path(dst_node, receiver)

        self.controller.end_session()


@register_strategy("SEACACHE")
class SEACACHE(Strategy):
    """
    SEACACHE routing strategy is cache architecture for SEANet name resolution records
    Just like LCE, but driven by source of the request. If a cache miss happened, source will determine what to cache.
    """

    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)
        self.view.topology().dump_topology_info()
        self.view.topology().gen_topo_file()
        self.alpha = kwargs.get("alpha", 0.1)  # Space occupancy limit of switch's cache
        self.beta = 0.5  # A hyperparameter used to adjust the contribution of historical TTL
        self.k = kwargs.get("k", 1000)  # A hyperparameter used to set the maximum number of recommend records.
        self.rec_method = kwargs.get("rec_method", "random")

    def process_event(self, time, receiver, content, log, **kwargs):
        """
        ===== Main process of SEANet cache strategy =====
        kwargs:
            size: content size
        """
        # get all required data
        source = self.view.content_source(content)
        path = self.view.shortest_path(receiver, source)
        # Route requests to original source and queries caches on the path
        self.controller.start_session(time, receiver, content, log, **kwargs)
        serving_node = None
        for u, v in path_links(path):
            self.controller.forward_request_hop(u, v)
            if self.view.has_cache(v):
                if self.controller.get_content(v):
                    serving_node = v
                    break
            # No cache hits, get content from source
            self.controller.get_content(v)
            serving_node = v
        # Return content
        path = list(reversed(self.view.shortest_path(receiver, serving_node)))
        for u, v in path_links(path):
            self.controller.forward_content_hop(u, v)
            if self.view.has_cache(v):
                # insert content
                self.controller.put_content(v, ttl=self.view.get_content_ttl(v, content))

                # TODO 2023/4/22: pre-caching additional content to node v
                # * ==== first step: get the candidate recommendation list [(c1,v1),(c2,v2),...]
                cand_rec_l = self.view.get_related_content(content, v, serving_node, k=self.k, method=self.rec_method,
                                                           index=kwargs.get("index", -1), tm=time)
                # print(cand_rec_l[:5])

                # * ==== second step: get Numbers of incrementally distributed caches use available cache size of sw.
                ava_size = self.view.get_available_cache_size(v)
                # content_pop = sigmoid(self.view.get_content_freq(content))
                content_pop = self.view.get_content_pop(content)
                cand_size = int(self.alpha * ava_size * content_pop)
                # print("cand_rec_len: ", len(cand_rec_l), "available size: ", ava_size, "content_pop: ", content_pop, "cand_size: ", cand_size)

                # * ==== third step: get the TTL value of candidate cache records.
                cand_rec_l.sort(key=lambda x: x[1], reverse=True)
                cand_rec_l = cand_rec_l[:int(cand_size)]
                out_rec_l = []
                for c, va in cand_rec_l:
                    t = self.beta * self.view.get_content_ttl(v, c) + \
                        (1 - self.beta) * va * content_pop * self.view.get_default_ttl()
                    out_rec_l.append((c, t))
                # print("out_rec_l: ", len(out_rec_l))

                # * ==== fourth step: insert the output cache records into the cache of sw.
                for c, t in out_rec_l:
                    self.controller.put_content(v, content=c, ttl=t)

        # cal free space ration each 10 sec
        if time % 10 == 0:
            self.controller.cal_free_space_ratio(time)

        self.controller.end_session()
        # print(Sim_T.get_sim_time())


@register_strategy("SEALOC")
class SEALOC(Strategy):
    """
    SEANet locator selection strategy, simulation in one controller domain.
    """

    def __init__(self, view, controller, **kwargs):
        super().__init__(view, controller)
        self.alpha = kwargs.get("alpha", 0.1)
        self.view.topology().dump_topology_info()
        self.view.topology().gen_topo_file()
        self.method = kwargs.get("method", "main")
        # set weight_link for each web matrix, [capacity, latency, packet loss]
        self.weight_link = [0.5, 0.5, 0] if "weight_link" not in kwargs else kwargs["weight_link"]
        self.weight = [0.5, 0.5] if "weight" not in kwargs else kwargs["weight"]

    def G(self, dst_freq: dict, link_stats: dict, locs: list) -> list:
        """
        The function G is used to calculate the utility value of each location.
        return a pdf for locations
        """
        final_stats = []
        for loc in locs:
            final_stats.append(self.weight[0] * dst_freq[loc] + self.weight[1] * link_stats[loc])
        # turn to pdf
        final_stats = [x / sum(final_stats) for x in final_stats]
        return final_stats

    def process_event(self, time, receiver, content, log, **kwargs):
        """
        ===== Main process of SEANet locator selection strategy =====
        kwargs:
            size: request size
        """
        req_size = kwargs.get("size", 1)
        duration = kwargs.get("duration", 1)
        self.controller.start_session(time, receiver, content, log, **kwargs)
        acc_switch = self.view.get_access_switch(receiver)
        self.controller.forward_request_path(receiver, acc_switch)
        self.controller.packet_in(content)

        # ! 1. resolution module: get all locations of content
        locs = self.view.content_locations(content)

        # ! 2. frequency module: get dst_frequency for each location
        dst_frequency = {loc: self.view.get_dst_freq(loc) for loc in locs}

        # ! 3. statistic module: get the next hop link status for each location
        link_status = {}
        loc2hop = {}
        link_delays, link_capacities, link_losses = {}, {}, {}
        for loc in locs:
            next_hop_link = path_links(self.view.shortest_path(acc_switch, loc))[0]
            loc2hop[loc] = next_hop_link
            if next_hop_link in link_capacities:
                continue
            # get the link status
            link_capacities[next_hop_link] = max(self.view.link_capacity_avail(*next_hop_link, time), 0)
            link_delays[next_hop_link] = self.view.link_delay(*next_hop_link)
            link_losses[next_hop_link] = self.view.link_loss_rate(*next_hop_link)
        # normalize the link_delays, link_capacities, link_losses
        normalize(link_delays)
        normalize(link_capacities)
        normalize(link_losses)
        # calculate the link status
        for loc in locs:
            next_hop_link = loc2hop[loc]
            link_status[loc] = self.weight_link[0] * link_capacities[next_hop_link] + \
                               self.weight_link[1] * link_delays[next_hop_link] + \
                               self.weight_link[2] * link_losses[next_hop_link]

        # ! 4. select the best location use the above information
        locs_l = sorted(locs)
        if self.method == "main":
            # get the location with the adaptive address selection method
            target_loc = random.choices(locs_l, weights=self.G(dst_frequency, link_status, locs_l))[0]
        elif self.method == "first":
            target_loc = locs_l[0]
        elif self.method == "onlyfreq":
            # Select the minimum destination frequency location
            target_loc = min(dst_frequency, key=dst_frequency.get)
        elif self.method == "random":
            target_loc = random.choice(locs_l)
        elif self.method == "onlystates":
            # get the location with the highest link_status
            target_loc = max(link_status, key=link_status.get)
        else:
            raise ValueError("method should be one of ['main', 'ecmp', 'onlystates', 'onlyfreq', 'random']")

        # ! 5. route the request to the selected location and get content back
        self.controller.forward_request_path(acc_switch, target_loc)
        self.controller.get_content(target_loc)
        self.controller.forward_content_path(target_loc, receiver)

        # ! 6. update the status of the link, update the dst_frequency
        target_link = loc2hop[target_loc]
        # At time, the capacity usage on the link increases by req_size
        self.controller.update_link_capacity_used(*target_link, time, req_size)
        # At time + duration, the capacity usage on the link decreases by req_size
        self.controller.update_link_capacity_used(*target_link, time + duration, -req_size)
        self.controller.update_dst_freq(target_loc, Sim_T.get_sim_time())
        self.controller.end_session()
