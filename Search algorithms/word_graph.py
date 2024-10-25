from collections import defaultdict
import Levenshtein
    
def load_dictionary(file_path):
    #load dictionary from file
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        words = [word.strip().lower() for row in csv_reader for word in row if word.strip()]

    return words

def build_word_graph(words):
    #build word graph, connecting words with edit distance of 1
    graph = {word: [] for word in words} 
    for word in words:
        for other_word in words:
            if word != other_word and Levenshtein.distance(word, other_word) == 1:
                graph[word].append(other_word)

    return graph

def get_words_of_length(words, length):
    #get list of words with specified length
    return [word for word in words if len(word) == length]