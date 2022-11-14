# Tests for dht
from icarus.utils.dht_chord.dht import *
import random


if __name__ == '__main__':
    d = DHT(10)
    print("Number of nodes in the system: ", d.get_num_nodes())
    # print("Finger table of node 0: ", d._startNode.fingerTable)
    # Add nodes
    for i in range(1, 1000, 10):
        d.join(NodeDHT(i))
    print("Number of nodes in the system after join: ", d.get_num_nodes())

    # d.updateAllFingerTables()

    # add key-value pairs
    for i in range(100):
        d.store(i, "hello" + str(i))

    # search for a key
    key = 3
    print("Searching for key: ", d.lookup(key))
    nxt = d.find_next_direct_node(key, d.get_start_node())
    print("find next node: ", nxt.ID)
    key = 1000
    print("Searching for key: ", d.lookup(key))



