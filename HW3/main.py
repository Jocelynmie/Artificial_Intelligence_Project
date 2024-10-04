from word_graph import build_word_graph, get_words_of_length
from search_algorithms import (
    find_edit_path_bfs, 
    find_edit_path_dfs, 
    find_edit_path_iterative_deepening, 
    find_edit_path_A_star_search
)
import csv


def load_dictionary(file_path):
    #Reading in dictionary and limits word length
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        return [word.strip().lower() for row in csv_reader for word in row if word.strip()]

def main():
    file_path = '/Users/yangwenyu/Desktop/CS5100 /HW3/database.csv'
    dictionary = load_dictionary(file_path)
    
    #gets start and end words from user
    start_word = input("Enter the start word (length should be 5): ")
    end_word = input("Enter the end word (length should be 5): ")
    
    while len(start_word) != len(end_word):
        print("Start and end words must be the same length.")
        start_word = input("Enter the start word: ")
        end_word = input("Enter the end word: ")
        
    #reads in dictionary and limits word length
    words = get_words_of_length(dictionary, len(start_word))
    #makes adjacency list
    graph = build_word_graph(words)

    print("BFS:", find_edit_path_bfs(graph, start_word, end_word))
    print("DFS:", find_edit_path_dfs(graph, start_word, end_word))
    print("Iterative Deepening:", find_edit_path_iterative_deepening(graph, start_word, end_word))
    print("A* Search:", find_edit_path_A_star_search(graph, start_word, end_word))

if __name__ == "__main__":
    main()