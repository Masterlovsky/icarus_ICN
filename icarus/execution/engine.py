"""This module implements the simulation engine.

The simulation engine, given the parameters according to which a single
experiments needs to be run, instantiates all the required classes and executes
the experiment by iterating through the event provided by an event generator
and providing them to a strategy instance.
"""
from tqdm import tqdm

from icarus.execution import (
    NetworkModel,
    SEANRSModel,
    MDHTModel,
    NetworkView,
    NetworkController,
    CollectorProxy,
    simulation_time,
)
from icarus.registry import DATA_COLLECTOR, STRATEGY
from icarus.scenarios import SEANRS_Topology

__all__ = ["exec_experiment"]


def exec_experiment(topology, workload, netconf, strategy, cache_policy, collectors):
    """Execute the simulation of a specific scenario.

    Parameters
    ----------
    topology : Topology
        The FNSS Topology object modelling the network topology on which
        experiments are run.
    workload : iterable
        An iterable object whose elements are (time, event) tuples, where time
        is a float type indicating the timestamp of the event to be executed
        and event is a dictionary storing all the attributes of the event to
        execute
    netconf : dict
        Dictionary of attributes to inizialize the network model
    strategy : tree
        Strategy definition. It is tree describing the name of the strategy
        to use and a list of initialization attributes
    cache_policy : tree
        Cache policy definition. It is tree describing the name of the cache
        policy to use and a list of initialization attributes
    collectors: dict
        The collectors to be used. It is a dictionary in which keys are the
        names of collectors to use and values are dictionaries of attributes
        for the collector they refer to.

    Returns
    -------
    results : Tree
        A tree with the aggregated simulation results from all collectors
    """
    strategy_name = strategy["name"]
    # *  ---- 1. create Network Model ----
    if isinstance(topology, SEANRS_Topology):
        if strategy_name == "MDHT":
            model = MDHTModel(topology, cache_policy, workload, **netconf)
        else:
            model = SEANRSModel(topology, cache_policy, workload, **netconf)
    else:
        model = NetworkModel(topology, cache_policy, workload, **netconf)

    # * ---- 2. create Network View ----
    view = NetworkView(model)

    # * ---- 3. create Network Controller ----
    controller = NetworkController(model)

    # * ---- 4. attach Collectors to controller ----
    collectors_inst = [
        DATA_COLLECTOR[name](view, **params) for name, params in collectors.items()
    ]
    collector = CollectorProxy(view, collectors_inst)
    controller.attach_collector(collector)

    # * ---- 5. Use strategy to process event ----
    strategy_args = {k: v for k, v in strategy.items() if k != "name"}
    strategy_inst = STRATEGY[strategy_name](view, controller, **strategy_args)

    for time, event in tqdm(workload, desc="Processing request: "):
        simulation_time.Sim_T.set_sim_time(time)
        strategy_inst.process_event(time, **event)
    return collector.results()
