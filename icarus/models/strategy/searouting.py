"""Implementations of SEANet strategies"""
import random

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy
import logging

__all__ = [
    "SEANRS",
]

logger = logging.getLogger("SEANRS-strategy")


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
            # todo: this will be changed to the content_location
            return self.view.content_source(content)
        # Check whether the content is in the sdn controller
        self.controller.packet_in(content)
        dst_n = self.view.get_dst_from_controller(
            self.view.get_asn(sw), self.view.get_ctrln(sw), content)
        if dst_n:
            self.controller.get_content(dst_n)
            # Update the cache of the switch
            if self.view.has_cache(sw):
                self.controller.put_content(sw)
            return dst_n

    def resolve_bgp(self, bgn, inter=False):
        """Resolve a content request in the bgp domain

        Parameters
        ----------
        bgn : Node
            The bgp node dealing with the request
        inter: bool
            Whether the request is from the control-domain or manage-domain

        Returns
        -------
        node_list : [Node]
            The valid access node list to route the request to
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
        nearest_sw_list = []  # nearest_sw: [1, 4, ...]
        if mcf.contains(content):
            # if the content fingerprint is in the mcf, get the set mask from the mcf
            area, idx_l = mcf.decode_mask(mcf.get(content))
            if area == 'bit':
                for idx in idx_l:
                    acc_sw.append(self.view.get_switches_in_ctrl(self.view.get_asn(bgn), idx))
                # get the nearest switch for each valid controller domain
                nearest_sw_list = [get_nearest_sw(bgn, sw_l) for sw_l in acc_sw]
            else:
                if inter:
                    return
                # Directly goto the next inter-bgn to resolve the content
                bgn_next = random.choice(list(self.view.get_bgns_in_as(idx_l[0])))
                nearest_sw_list = self.resolve_bgp(bgn_next, True)
                if not nearest_sw_list:
                    logger.error('Perhaps false positive happens, No valid access node found in the inter-domain bgn.')
                    # todo: return or hash the content fingerprint to find another bgn?

        return nearest_sw_list

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        """
        Process a content request event
        ===== Main process of SEANet routing strategy =====
        """
        # route content to the access switch
        acc_switch = self.view.get_access_switch(receiver)
        self.controller.start_session(time, receiver, content, log)
        # route the request from receiver to access switch
        self.controller.forward_request_path(receiver, acc_switch)
        # if the content is resolved in the ctrl domain, get content and return
        dst_node = self.resolve_ctrl(acc_switch)
        if dst_node:
            self.controller.forward_request_path(acc_switch, dst_node)
            # self.controller.get_content(dst_node)
            # forward content back to receiver
            self.controller.forward_content_path(dst_node, receiver)
            self.controller.end_session(success=True)
            return
        # if the content is not resolved in the ctrl domain, get the content from the manage-domain
        # get bgn of acc_switch
        bgns: set = self.view.get_bgns_in_as(self.view.get_asn(acc_switch))
        # todo: choose a bgn from bgns set, now choose the first one
        bgn = random.choice(list(bgns))
        # route the request to bgn
        self.controller.forward_request_path(acc_switch, bgn)
        # if the content is resolved in the manage-domain, get acc-switch
        nearest_sw_list = self.resolve_bgp(bgn)
        if nearest_sw_list:
            for sw in nearest_sw_list:
                self.controller.forward_request_path(bgn, sw)
                dst_node = self.resolve_ctrl(sw)
                if dst_node:
                    # self.controller.get_content(dst_node)
                    self.controller.forward_request_path(sw, dst_node)
                    break
                # if the content is not resolved in the foreign ctrl domain, false positive happens, route back to bgn
                self.controller.forward_request_path(sw, bgn)

        if dst_node:
            self.controller.forward_content_path(dst_node, receiver)
            self.controller.end_session(success=True)
            return

        # if the content is not resolved in the manage-domain, hash the content fingerprint to find inter-domain bgn
        inter_bgn = self.view.get_bgn_conhash(str(content))
        self.controller.forward_request_path(bgn, inter_bgn)
        nearest_sw_list = self.resolve_bgp(inter_bgn)
        if nearest_sw_list:
            for sw in nearest_sw_list:
                self.controller.forward_request_path(inter_bgn, sw)
                dst_node = self.resolve_ctrl(sw)
                if dst_node:
                    self.controller.forward_request_path(sw, dst_node)
                    break
                # if the content is not resolved in the foreign ctrl domain, false positive happens, route back to bgn
                self.controller.forward_request_path(sw, inter_bgn)
        else:
            logger.warning('No valid access node found in the inter-domain bgn.')

        self.controller.end_session()
