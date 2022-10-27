# testing the consistent hashing methods
from icarus.scenarios.topology import SEANRS_Topology
import fnss
from icarus.uhashring import HashRing
import mmh3

class TestNode(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

if __name__== '__main__':
    ring = HashRing(hash_fn=mmh3.hash)
    ring.add_node(TestNode('node1'), conf={'weight': 1, 'vnode': 100, 'instance': 1})
    ring.add_node(TestNode('node2'), conf={'weight': 2, 'vnode': 100, 'instance': 1})
    ring.add_node(TestNode('node3'), conf={'weight': 1, 'vnode': 100})
    n1 = n2 = n3 = 0
    for k in range(100):
        node = ring.get_node('key%s' % k)
        # calculate
        if node.name == 'node1':
            n1 += 1
        elif node.name == 'node2':
            n2 += 1
        elif node.name == 'node3':
            n3 += 1
    print(n1, n2, n3)
    print("===============================")
    print(ring.get_node_instance('111'))
    print(ring.get_node('111'))
    print(ring.get_node_weight('111'))
    ring.add_node(TestNode('node4'), conf={'weight': 1, 'vnode': 100, 'instance': 1})
    print(ring.get_node('111'))
    print(ring.get_node('111'))
    print("===============================")
    conhashring = HashRing(hash_fn=mmh3.hash)
    topology = SEANRS_Topology()
    topology.add_nodes_from(range(3))
    topology.add_edges_from([(0, 1), (1, 2), (2, 0)])
    for n in topology.nodes():
        conhashring.add_node(n)
    print(conhashring.get_node('111'))
    print(conhashring.get_node('123'))
    print(conhashring.get_node('3332'))