import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque

def load_maze_from_file(filename="maze.txt"):
    maze = []
    with open(filename, 'r') as file:
        for line in file:
            maze.append(list(map(int, line.strip().split())))
    return maze

def bfs(maze, start, goals):
    rows, cols = len(maze), len(maze[0])
    visited = np.full((rows, cols), False)
    parent = np.full((rows, cols), None, dtype=object)
    queue = deque()
    goals_set = set(goals)

    sx, sy = start
    visited[sx][sy] = True
    queue.append(start)

    goal_reached = None

    while queue:
        x, y = queue.popleft()

        if (x, y) in goals_set:
            goal_reached = (x, y)
            break

        # Arah: Atas, Kanan, Bawah, Kiri
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if not visited[nx][ny] and maze[nx][ny] != -1:
                    visited[nx][ny] = True
                    parent[nx][ny] = (x, y)
                    queue.append((nx, ny))

    # Jika tujuan tercapai, bangun jalur dari parent
    if goal_reached:
        path = []
        x, y = goal_reached
        while (x, y) != start:
            path.append((x, y))
            x, y = parent[x][y]
        path.append(start)
        path.reverse()
        return path, goal_reached
    else:
        return None, None

def visualize_maze(maze, path, start, goals, goal_reached):
    maze_display = np.zeros_like(maze, dtype=int)  # Inisialisasi dengan 0 (jalan)

    # Mengubah dinding (-1) menjadi 1 untuk visualisasi (hitam)
    maze_display[np.array(maze) == -1] = 1  # Dinding

    # Menandai jalur yang ditemukan
    if path:
        for position in path:
            x, y = position
            maze_display[x][y] = 2  # Jalur

    # Menandai titik awal
    sx, sy = start
    maze_display[sx][sy] = 3  # Titik awal (merah)

    # Menandai semua titik tujuan
    for gx, gy in goals:
        if (gx, gy) == goal_reached:
            maze_display[gx][gy] = 4  # Tujuan yang dicapai (hijau)
        else:
            maze_display[gx][gy] = 5  # Tujuan lainnya (kuning)

    # Membuat custom colormap dengan warna yang disesuaikan
    cmap = ListedColormap(['white', 'black', 'lightblue', 'red', 'green', 'yellow'])

    plt.figure(figsize=(8,8))
    plt.imshow(maze_display, cmap=cmap)
    plt.title('Visualisasi Maze dengan BFS (Dua Titik Tujuan)')
    plt.axis('off')
    plt.show()

def main():
    # Membaca maze dari file
    maze = load_maze_from_file("Rute/maze.txt")

    # ================================================================>>>>>>>>>>>>
    start = (20, 12)  # Titik awal
    goals = [(0, 8), (3, 16)]  # Daftar titik tujuan

    path, goal_reached = bfs(maze, start, goals)
    if path:
        print("Jalur ditemukan menuju", goal_reached)
        print("Panjang jalur:", len(path) - 1)
        print("Jalur:", path)
        visualize_maze(maze, path, start, goals, goal_reached)
    else:
        print("Tidak ada jalur yang ditemukan.")

if __name__ == "__main__":
    main()
