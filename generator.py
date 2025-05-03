import numpy as np
import random


def generate_maze(rows=15, cols=15):
    maze = np.ones((rows * 2 + 1, cols * 2 + 1), dtype=int)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def carve(x, y):
        maze[2 * x + 1, 2 * y + 1] = 0  # cell open
        dirs = directions.copy()
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[2 * nx + 1, 2 * ny + 1] == 1:
                # Enlever le mur entre les deux cellules
                maze[x + nx + 1, y + ny + 1] = 0
                carve(nx, ny)

    carve(0, 0)  
    return maze
