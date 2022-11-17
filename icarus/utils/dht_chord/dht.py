# A Distributed Hash Table implementation

class NodeDHT:
    def __init__(self, ID, nxt=None, prev=None):
        self.ID = ID
        self.data = dict()
        self.prev = prev
        self.fingerTable = [nxt]

    # Update the finger table of this node when necessary
    def update_finger_table(self, dht, k):
        del self.fingerTable[1:]
        for i in range(1, k):
            self.fingerTable.append(dht.find_node(dht.get_start_node(), self.ID + 2 ** i))

    def __repr__(self):
        return str(self.ID)

    def __eq__(self, other):
        return self.ID == other.ID


class DHT:
    # The total number of IDs available in the DHT is 2 ** k
    def __init__(self, k: int, start: int = 0):
        self._k = k
        self._size = 2 ** k
        self._startNode = NodeDHT(start, k)
        self._startNode.fingerTable[0] = self._startNode
        self._startNode.prev = self._startNode
        self._startNode.update_finger_table(self, k)

    def get_start_node(self):
        return self._startNode

    # Hash function used to get the ID
    def get_hash_id(self, key) -> int:
        return hash(key) % self._size

    # Get distance between two IDs
    def distance(self, n1: int, n2: int) -> int:
        if n1 == n2:
            return 0
        if n1 < n2:
            return n2 - n1
        return self._size - n1 + n2

    # Get number of nodes in the system
    def get_num_nodes(self) -> int:
        if self.get_start_node() is None:
            return 0
        node = self._startNode
        n = 1
        while node.fingerTable[0] != self._startNode:
            n = n + 1
            node = node.fingerTable[0]
        return n

    def find_next_direct_node(self, key, start):
        hashid = self.get_hash_id(key)
        if start.ID == hashid:
            return start
        if self.distance(start.ID, hashid) <= self.distance(start.fingerTable[0].ID, hashid):
            return start.fingerTable[0]
        tabsize = len(start.fingerTable)
        i = 0
        next_node = start.fingerTable[-1]
        while i < tabsize - 1:
            if self.distance(start.fingerTable[i].ID, hashid) < self.distance(start.fingerTable[i + 1].ID, hashid):
                next_node = start.fingerTable[i]
                break
            i = i + 1
        return next_node

    # Find the node responsible for the key
    def find_node(self, start, key):
        hashid = self.get_hash_id(key)
        curr = start
        num_jumps = 0
        while True:
            if curr.ID == hashid:
                # print("number of jumps: ", numJumps)
                return curr
            if self.distance(curr.ID, hashid) <= self.distance(curr.fingerTable[0].ID, hashid):
                # print("number of jumps: ", numJumps)
                return curr.fingerTable[0]
            tabsize = len(curr.fingerTable)
            i = 0
            next_node = curr.fingerTable[-1]
            while i < tabsize - 1:
                if self.distance(curr.fingerTable[i].ID, hashid) < self.distance(curr.fingerTable[i + 1].ID, hashid):
                    next_node = curr.fingerTable[i]
                    break
                i = i + 1
            curr = next_node
            num_jumps += 1

    # Look up a key in the DHT, return the value if found, None otherwise
    def lookup(self, key, start=None):
        if start is None:
            start = self._startNode
        node_for_key = self.find_node(start, key)
        return DHT.get_value_from_node(node_for_key, key)

    @staticmethod
    def get_value_from_node(node, key):
        if key in node.data:
            return node.data[key]
        return None

    # Store a key-value pair in the DHT
    def store(self, key, value, start=None):
        if start is None:
            start = self._startNode
        node_for_key = self.find_node(start, key)
        node_for_key.data[key] = value

    # When new node joins the system
    def join(self, newNode):
        # Find the node before which the new node should be inserted
        orig_node = self.find_node(self._startNode, newNode.ID)

        # print(origNode.ID, "  ", newNode.ID)
        # If there is a node with the same id, decline the join request for now
        if orig_node.ID == newNode.ID:
            # print("There is already a node with the same id:{}!".format(newNode.ID))
            return False

        # Copy the key-value pairs that will belong to the new node after
        # the node is inserted in the system
        for key in orig_node.data:
            hashid = self.get_hash_id(key)
            if self.distance(hashid, newNode.ID) < self.distance(hashid, orig_node.ID):
                newNode.data[key] = orig_node.data[key]

        # Update the prev and next pointers
        prev_node = orig_node.prev
        newNode.fingerTable[0] = orig_node
        newNode.prev = prev_node
        orig_node.prev = newNode
        prev_node.fingerTable[0] = newNode

        # Set up finger table of the new node
        newNode.update_finger_table(self, self._k)

        # Delete keys that have been moved to new node
        for key in list(orig_node.data.keys()):
            hashid = self.get_hash_id(key)
            if self.distance(hashid, newNode.ID) < self.distance(hashid, orig_node.ID):
                del orig_node.data[key]

        return True

    def leave(self, node):
        # Copy all its key-value pairs to its successor in the system
        for k, v in node.data.items():
            node.fingerTable[0].data[k] = v
        # If this node is the only node in the system.
        if node.fingerTable[0] == node:
            self._startNode = None
        else:
            node.prev.fingerTable[0] = node.fingerTable[0]
            node.fingerTable[0] = node.prev
            # If this deleted node was an entry point to the system, we
            # need to choose another entry point. Simply choose its successor
            if self._startNode == node:
                self._startNode = node.fingerTable[0]

    def update_all_finger_tables(self):
        self._startNode.update_finger_table(self, self._k)
        curr = self._startNode.fingerTable[0]
        while curr != self._startNode:
            curr.update_finger_table(self, self._k)
            curr = curr.fingerTable[0]
