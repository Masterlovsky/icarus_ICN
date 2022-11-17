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
        d.store(str(i), "hello" + str(i))

    d.update_all_finger_tables()
    print("dht[{}]: {}".format(0, d.get_start_node().fingerTable))
    # search for a key
    key = "32"
    print("Searching for key {} ,result: {}".format(key, d.lookup(key)))
    nxt = d.get_start_node()
    res = d.find_node(nxt, key)
    print("res_node is {}".format(res.ID))
    while nxt != res:
        print("find next node: {}, content:{}".format(nxt.ID, nxt.data))
        nxt = d.find_next_direct_node(key, nxt)

    # nxt = d.find_next_direct_node(key, nxt)
    print("Finally find next node: {}, value:{}".format(nxt.ID, d.get_value_from_node(nxt, key)))
    key = "1000"
    print("Searching for key: ", d.lookup(key))
