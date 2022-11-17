"""Implementations of MDHT strategies"""
import random

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy
import logging

__all__ = [
    "MDHT",
]

logger = logging.getLogger("MDHT-strategy")


@register_strategy("MDHT")
class MDHT(Strategy):
    """
    This strategy implements the MDHT routing strategy.
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
        self.dhts = self.view.get_dhts()

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        self.controller.start_session(time, receiver, content, log)
        # ! >>> 1. receiver send request to access switch
        acc_sw = self.view.get_access_switch(receiver)
        self.controller.forward_request_path(receiver, acc_sw)
        # ! >>> 2. access switch check dhts[1], if success, get the dst
        as_id = str(self.view.get_asn(acc_sw))
        ctrl_id = str(self.view.get_ctrln(acc_sw))
        dht = self.dhts[1][str(as_id) + "@" + str(ctrl_id)]
        res_node_l1_dht = dht.find_node(dht.get_start_node(), str(content))
        # print("resolver_node_l1: {}, content: {}".format(resolver_node_l1, content))
        res_node_l1_topo = self.view.dn2tn("l1@" + as_id + "@" + ctrl_id + "@" + str(res_node_l1_dht.ID))
        self.controller.forward_request_path(acc_sw, res_node_l1_topo)
        dst = dht.get_value_from_node(res_node_l1_dht, str(content))
        if dst:
            # print("Get content:{} from dht-level-1, size: {}".format(content, dht.get_num_nodes()))
            self.controller.forward_request_path(res_node_l1_topo, dst)
            self.controller.forward_content_path(dst, receiver)
            self.controller.end_session()
            return
        # !  >>> 2.1 if fail, check dhts[2], if success, get the dst
        dht = self.dhts[2][str(as_id)]
        res_node_l2_dht = dht.find_node(dht.get_start_node(), str(content))
        res_node_l2_topo = self.view.dn2tn("l2@" + as_id + "@" + str(res_node_l2_dht.ID))
        self.controller.forward_request_path(res_node_l1_topo, res_node_l2_topo)
        dst = dht.get_value_from_node(res_node_l2_dht, str(content))
        if dst:
            # print("Get content:{} from dht-level-2, size: {}".format(content, dht.get_num_nodes()))
            self.controller.forward_request_path(res_node_l2_topo, dst)
            self.controller.forward_content_path(dst, receiver)
            self.controller.end_session()
            return
        # !  >>> 2.2 if fail, check dhts[3] hop by hop, if success, get the dst
        dht = self.dhts[3]["G"]
        res_node_l3_dht = dht.find_node(dht.get_start_node(), str(content))
        cur_node = dht.find_next_direct_node(str(content), dht.get_start_node())
        cur_topo = self.view.dn2tn("l3@" + str(cur_node.ID))
        self.controller.forward_request_path(res_node_l2_topo, cur_topo)
        while cur_node and cur_node != res_node_l3_dht:
            # print("Get content:{} from dht-level-3, cur_node: {}, resolve_node: {}"
            #       .format(content, cur_node.ID, res_node_l3_dht.ID))
            next_node = dht.find_next_direct_node(str(content), cur_node)
            next_topo = self.view.dn2tn("l3@" + str(next_node.ID))
            self.controller.forward_request_path(cur_topo, next_topo)
            cur_node = next_node
            cur_topo = next_topo
        dst = dht.get_value_from_node(res_node_l3_dht, str(content))
        if dst:
            self.controller.forward_request_path(cur_topo, dst)
            self.controller.forward_content_path(dst, receiver)
            self.controller.end_session()
            return
        else:
            # !  >>> 2.3 if fail, no content, return
            self.controller.end_session()
            logger.warning("No content found for content {%s} in the global view.", content)

        return
