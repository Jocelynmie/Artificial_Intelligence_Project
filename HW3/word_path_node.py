class WordPathNode:
    def __init__(self, word, parent=None):
        self.word = word
        self.parent = parent
    
    def __eq__(self, other):
        return self.word == other.word

    def __lt__(self, other):
        return self.word < other.word
    
    def __hash__(self):
        return hash(self.word)
    
    def get_path_to_root(self):
        path = []
        current = self
        while current:
            path.append(current.word)
            current = current.parent
        return path[::-1]