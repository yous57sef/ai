# from collections import deque

# def bfs_solver(maze, start=(1, 1), goal=None):
#     if goal is None:
#         goal = (maze.shape[0] - 2, maze.shape[1] - 2)

#     queue = deque([start])
#     visited = set()
#     parent = {}

#     while queue:
#         current = queue.popleft()
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
#                 queue.append((nx, ny))
#                 parent[(nx, ny)] = current

#     # Reconstruire le chemin
#     path = []
#     node = goal
#     while node != start:
#         path.append(node)
#         node = parent.get(node)
#         if node is None:
#             return [], visited  # aucun chemin
#     path.append(start)
#     path.reverse()
#     return path, visited

from collections import deque, defaultdict

def bfs_solver(maze):
    rows, cols = maze.shape
    start = (1, 1)
    end = (rows - 2, cols - 2)
    visited = []
    parents = {}
    queue = deque([start])
    visited.append(start)
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == end:
            break
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                maze[nx, ny] == 0 and (nx, ny) not in visited):
                queue.append((nx, ny))
                visited.append((nx, ny))
                parents[(nx, ny)] = (x, y)
    
    # Reconstruct path
    path = []
    current = end
    while current != start:
        path.append(current)
        current = parents.get(current, start)
    path.append(start)
    path.reverse()
    
    return path, visited, parents