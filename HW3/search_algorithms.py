from collections import deque
import heapq
from word_path_node import WordPathNode

def find_edit_path_bfs(graph, start_word, end_word):
    queue = deque([WordPathNode(start_word)])
    visited = set()

    while queue:
        current_node = queue.popleft()
        current_word = current_node.word

        if current_word == end_word:
            return current_node.get_path_to_root()

        if current_word in visited:
            continue

        visited.add(current_word)

        for neighbor in graph[current_word]:
            if neighbor not in visited:
                queue.append(WordPathNode(neighbor, current_node))

    return None

def find_edit_path_dfs(graph, start_word, end_word):
    stack = [WordPathNode(start_word)]
    visited = set()

    while stack:
        current_node = stack.pop()
        current_word = current_node.word

        if current_word == end_word:
            return current_node.get_path_to_root()

        if current_word in visited:
            continue

        visited.add(current_word)

        for neighbor in graph[current_word]:
            if neighbor not in visited:
                stack.append(WordPathNode(neighbor, current_node))

    return None

def find_edit_path_iterative_deepening(graph, start_word, end_word):
    def dfs_with_depth_limit(node, depth):
        if depth == 0 and node.word == end_word:
            return node.get_path_to_root()
        if depth > 0:
            for neighbor in graph[node.word]:
                child_node = WordPathNode(neighbor, node)
                result = dfs_with_depth_limit(child_node, depth - 1)
                if result:
                    return result
        return None

    max_depth = len(graph)
    for depth in range(max_depth):
        result = dfs_with_depth_limit(WordPathNode(start_word), depth)
        if result:
            return result
    return None

def find_edit_path_A_star_search(graph, start_word, end_word):
    
    def heuristic(word):
        return sum(1 for a, b in zip(word, end_word) if a != b)

    start_node = WordPathNode(start_word)
    frontier = [(0 + heuristic(start_word), 0, start_node)]
    came_from = {}
    cost_so_far = {start_word: 0}

    while frontier:
        _, current_cost, current_node = heapq.heappop(frontier)
        current_word = current_node.word

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

    return None