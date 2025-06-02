# import heapq

# def manhattan(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def astar_solver(maze, start=(1, 1), goal=None):
#     if goal is None:
#         goal = (maze.shape[0] - 2, maze.shape[1] - 2)

#     open_set = []
#     heapq.heappush(open_set, (0, start))
#     came_from = {}
#     g_score = {start: 0}
#     visited = set()

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if current in visited:
#             continue
#         visited.add(current)

#         if current == goal:
#             break

#         x, y = current
#         for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
#             nx, ny = x + dx, y + dy
#             neighbor = (nx, ny)
#             if (0 <= nx < maze.shape[0] and
#                 0 <= ny < maze.shape[1] and
#                 maze[nx, ny] == 0):

#                 tentative_g = g_score[current] + 1
#                 if tentative_g < g_score.get(neighbor, float('inf')):
#                     came_from[neighbor] = current
#                     g_score[neighbor] = tentative_g
#                     f_score = tentative_g + manhattan(neighbor, goal)
#                     heapq.heappush(open_set, (f_score, neighbor))

#     # Reconstruire le chemin
#     path = []
#     node = goal
#     while node != start:
#         path.append(node)
#         node = came_from.get(node)
#         if node is None:
#             return [], visited
#     path.append(start)
#     path.reverse()
#     return path, visited

from heapq import heappush, heappop
from collections import defaultdict

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def astar_solver(maze):
    rows, cols = maze.shape
    start = (1, 1)
    end = (rows - 2, cols - 2)
    visited = []
    parents = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, end)}
    open_set = [(f_score[start], start)]
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    while open_set:
        _, current = heappop(open_set)
        
        if current not in visited:
            visited.append(current)
            
            if current == end:
                break
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if (0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0):
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        parents[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, end)
                        heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current != start:
        path.append(current)
        current = parents.get(current, start)
    path.append(start)
    path.reverse()
    
    return path, visited, parents