from collections import defaultdict
import Levenshtein
    
def load_dictionary(file_path):
    file_path = '/Users/yangwenyu/Desktop/CS5100 /HW3/database.csv'
    with open(file_path, 'r') as file:
        return [word.strip().lower() for word in file]

def build_word_graph(words):
    graph = defaultdict(list)
    for word in words:
        for other_word in words:
            if Levenshtein.distance(word, other_word, score_cutoff=1) == 1:
                graph[word].append(other_word)
    return graph

def get_words_of_length(words, length):
    return [word for word in words if len(word) == length]


