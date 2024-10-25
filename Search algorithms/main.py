from word_graph import build_word_graph, get_words_of_length
from collections import deque
from search_algorithms import (
    find_edit_path_bfs, 
    find_edit_path_dfs, 
    find_edit_path_iterative_deepening, 
    find_edit_path_A_star_search
)
import csv
def is_connected(graph, start, end):
    if start not in graph or end not in graph:
        print(f"Start word '{start}' or end word '{end}' not in graph.")
        return False
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        if node not in visited:
            visited.add(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    print(f"Path not found. Visited {len(visited)} nodes.")
    return False

def load_dictionary(file_path):
    #Reading in dictionary and limits word length
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        return [word.strip().lower() for row in csv_reader for word in row if word.strip()]



def main():
    file_path = '/Users/yangwenyu/Desktop/CS5100 /HW3/database.csv'
    dictionary = load_dictionary(file_path)
    
    start_word = input("Enter the start word (length should be 5): ")
    end_word = input("Enter the end word (length should be 5): ")
    
    while len(start_word) != len(end_word):
        print("Start and end words must be the same length.")
        start_word = input("Enter the start word: ")
        end_word = input("Enter the end word: ")
        
    words = get_words_of_length(dictionary, len(start_word))

    graph = build_word_graph(words)

  
    if start_word not in graph or end_word not in graph:
        print("Start word or end word not found in the graph.")
        return

    if not is_connected(graph, start_word, end_word):
        print("No path exists between start word and end word.")
        return


    print("BFS:", find_edit_path_bfs(graph, start_word, end_word))
    print("DFS:", find_edit_path_dfs(graph, start_word, end_word))
    print("Iterative Deepening:", find_edit_path_iterative_deepening(graph, start_word, end_word))
    print("A* Search:", find_edit_path_A_star_search(graph, start_word, end_word))

if __name__ == "__main__":
    main()