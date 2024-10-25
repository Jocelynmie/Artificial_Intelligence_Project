class WordPathNode:
    def __init__(self, word, parent=None):
        self.word = word
        self.parent = parent
    
    def __eq__(self, other):
        #compare if two nodes are equal
        return self.word == other.word

    def __lt__(self, other):
        #compare if one node is less than another
        return self.word < other.word
    
    def __hash__(self):
        #get hash value of the node
        return hash(self.word)
    
    def get_path_to_root(self):
        #get path from current node to root
        path = []
        current = self
        while current:
            path.append(current.word)
            current = current.parent
        return path[::-1]  #reverse path to go from root to current node