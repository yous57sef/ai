# def dfs_solver(maze, start=(1, 1), goal=None):
#     if goal is None:
#         goal = (maze.shape[0] - 2, maze.shape[1] - 2)

#     stack = [start]
#     visited = set()
#     parent = {}

#     while stack:
#         current = stack.pop()
#         if current in visited:
#             continue
#         visited.add(current)

#         if current == goal:
#             break

#         x, y = current
#         for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
#             nx, ny = x + dx, y + dy
#             if (0 <= nx < maze.shape[0] and
#                 0 <= ny < maze.shape[1] and
#                 maze[nx, ny] == 0 and
#                 (nx, ny) not in visited):
#                 stack.append((nx, ny))
#                 parent[(nx, ny)] = current

#     # Reconstruire le chemin
#     path = []
#     node = goal
#     while node != start:
#         path.append(node)
#         node = parent.get(node)
#         if node is None:
#             return [], visited
#     path.append(start)
#     path.reverse()
#     return path, visited

from collections import defaultdict

def dfs_solver(maze):
    rows, cols = maze.shape
    start = (1, 1)
    end = (rows - 2, cols - 2)
    visited = []
    path = []
    parents = {}
    stack = [(start, [start])]
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    while stack:
        (x, y), current_path = stack.pop()
        if (x, y) not in visited:
            visited.append((x, y))
            
            if (x, y) == end:
                path = current_path
                break
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < rows and 0 <= ny < cols and
                    maze[nx, ny] == 0 and (nx, ny) not in visited):
                    stack.append(((nx, ny), current_path + [(nx, ny)]))
                    parents[(nx, ny)] = (x, y)
    
    return path, visited, parents