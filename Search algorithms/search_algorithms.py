from collections import deque
import heapq
import sys
from word_path_node import WordPathNode

def find_edit_path_bfs(graph, start_word, end_word):
    #initialize queue and visited set
    queue = deque([WordPathNode(start_word)])
    visited = set()

    while queue:
        current_node = queue.popleft()
        current_word = current_node.word

        #found target word, return path
        if current_word == end_word:
            return current_node.get_path_to_root()

        #skip visited words
        if current_word in visited:
            continue

        visited.add(current_word)

        #check all neighbors
        for neighbor in graph[current_word]:
            if neighbor not in visited:
                queue.append(WordPathNode(neighbor, current_node))

    #no path found
    return None

def dfs_with_depth_control(graph, current_node, end_word, visited, curr_depth, max_depth):
    #exceeded max depth, return none
    if curr_depth > max_depth:
        return None
    #found target word, return path
    if current_node.word == end_word:
        return current_node.get_path_to_root()
    
    visited.add(current_node.word)
    for neighbor in graph[current_node.word]:
        if neighbor not in visited:
            next_node = WordPathNode(neighbor, current_node)
            result = dfs_with_depth_control(graph, next_node, end_word, visited, curr_depth + 1, max_depth)
            if result:
                return result
    visited.remove(current_node.word)
    return None

def find_edit_path_dfs(graph, start_word, end_word):
    visited = set()
    #set max depth to half of recursion limit
    max_depth = sys.getrecursionlimit() // 2
    return dfs_with_depth_control(graph, WordPathNode(start_word), end_word, visited, 0, max_depth)

def find_edit_path_iterative_deepening(graph, start_word, end_word):
    max_possible_depth = len(graph)
    for max_depth in range(max_possible_depth):
        visited = set()
        result = dfs_with_depth_control(graph, WordPathNode(start_word), end_word, visited, 0, max_depth)
        if result:
            return result
    return None

def find_edit_path_A_star_search(graph, start_word, end_word):
    def heuristic(word):
        #count how many letters are different
        return sum(1 for a, b in zip(word, end_word) if a != b)

    start_node = WordPathNode(start_word)
    frontier = [(0 + heuristic(start_word), 0, start_node)]
    came_from = {}
    cost_so_far = {start_word: 0}

    while frontier:
        _, current_cost, current_node = heapq.heappop(frontier)
        current_word = current_node.word

        #found target word, return path
        if current_word == end_word:
            return current_node.get_path_to_root()

        for next_word in graph[current_word]:
            new_cost = current_cost + 1
            if next_word not in cost_so_far or new_cost < cost_so_far[next_word]:
                cost_so_far[next_word] = new_cost
                priority = new_cost + heuristic(next_word)
                next_node = WordPathNode(next_word, current_node)
                heapq.heappush(frontier, (priority, new_cost, next_node))
                came_from[next_word] = current_word

    #no path found
    return None