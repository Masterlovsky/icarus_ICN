"""Implementations of SEANet strategies"""
import random

import networkx as nx

from icarus.registry import register_strategy
from icarus.util import inheritdoc, path_links

from .base import Strategy

__all__ = [
    "SEANRS",
]

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

    @inheritdoc(Strategy)
    def process_event(self, time, receiver, content, log):
        # get source of content
        source = self.view.content_source(content)
        # route content to the access switch
        acc_switch = self.view.get_access_switch(receiver)
        path = self.view.shortest_path(receiver, acc_switch)
        self.controller.start_session(time, receiver, content, log)
        # todo: SEANRS strategy


        self.controller.end_session()