import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import heapq

def load_maze_from_file(filename="maze.txt"):
    """Memuat maze dari file teks"""
    maze = []
    with open(filename, 'r') as file:
        for line in file:
            maze.append(list(map(int, line.strip().split())))
    return maze

def dijkstra(maze, goals):
    rows, cols = len(maze), len(maze[0])
    distances = np.full((rows, cols), np.inf)
    visited = np.zeros((rows, cols), dtype=bool)
    heap = []

    for goal in goals:
        gx, gy = goal
        distances[gx][gy] = 0
        heapq.heappush(heap, (0, (gx, gy)))

    while heap:
        current_distance, (x, y) = heapq.heappop(heap)

        if visited[x][y]:
            continue
        visited[x][y] = True

        # Arah: Atas, Kanan, Bawah, Kiri
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if maze[nx][ny] != -1 and not visited[nx][ny]:
                    cost = maze[nx][ny]  # Biaya sel tetangga
                    distance = current_distance + cost
                    if distance < distances[nx][ny]:
                        distances[nx][ny] = distance
                        heapq.heappush(heap, (distance, (nx, ny)))

    return distances

def ant_colony_optimization(maze, start, goals, num_ants=100, num_iterations=200, alpha=1, beta=2, evaporation_rate=0.3, Q=100):
    rows, cols = len(maze), len(maze[0])
    pheromone = np.ones((rows, cols))  # Matriks feromon

    goals_set = set(goals)

    heuristic = dijkstra(maze, goals)  # Menggunakan Dijkstra untuk heuristik

    # Cek apakah start terhubung ke salah satu tujuan
    if heuristic[start[0]][start[1]] == np.inf:
        print("Tidak ada jalur yang mungkin dari titik awal ke tujuan.")
        return None, None, None

    best_path = None
    best_cost = float('inf')
    goal_reached = None  # Menyimpan tujuan yang dicapai

    for iteration in range(num_iterations):
        all_paths = []
        for ant in range(num_ants):
            path = [start]
            visited = set()
            visited.add(start)
            current_position = start
            total_cost = maze[current_position[0]][current_position[1]]  # Biaya awal

            while current_position not in goals_set:
                x, y = current_position
                neighbors = []

                # Arah: Atas, Kanan, Bawah, Kiri
                directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                random.shuffle(directions)  # Mengacak urutan untuk diversifikasi

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < rows and 0 <= ny < cols:
                        if maze[nx][ny] != -1 and (nx, ny) not in visited:
                            neighbors.append((nx, ny))

                if not neighbors:
                    break  # Tidak ada tetangga yang tersedia, semut terjebak

                # Menghitung probabilitas pemilihan tetangga
                probabilities = []
                for nx, ny in neighbors:
                    tau = pheromone[nx][ny] ** alpha
                    h = heuristic[nx][ny]
                    # Penanganan heuristik
                    if h == np.inf:
                        eta = 0  # Tidak boleh dipilih
                    elif h == 0:
                        eta = 1e6  # Sangat dekat dengan tujuan
                    else:
                        eta = (1 / h) ** beta
                    probabilities.append(tau * eta)

                total = sum(probabilities)
                if total == 0 or not np.isfinite(total):
                    break  # Tidak ada jalur yang mungkin
                probabilities = [p / total for p in probabilities]

                # Memilih tetangga berdasarkan probabilitas
                next_index = random.choices(range(len(neighbors)), weights=probabilities)[0]
                next_position = neighbors[next_index]

                # Biaya pergerakan adalah bobot sel tujuan
                move_cost = maze[next_position[0]][next_position[1]]

                path.append(next_position)
                visited.add(next_position)
                current_position = next_position
                total_cost += move_cost

            if current_position in goals_set:
                all_paths.append((path, total_cost, current_position))
                if total_cost < best_cost:
                    best_path = path
                    best_cost = total_cost
                    goal_reached = current_position  # Menyimpan tujuan yang dicapai

        # Pembaruan feromon
        pheromone *= (1 - evaporation_rate)  # Evaporasi
        for path, cost, _ in all_paths:
            delta_pheromone = Q / cost
            for position in path:
                x, y = position
                pheromone[x][y] += delta_pheromone

        # Menampilkan progres setiap 10 iterasi
        if (iteration + 1) % 10 == 0:
            if best_cost == float('inf'):
                print(f"Iterasi {iteration + 1}/{num_iterations}, Jalur terbaik saat ini: belum ditemukan")
            else:
                print(f"Iterasi {iteration + 1}/{num_iterations}, Biaya terbaik saat ini: {best_cost} menuju {goal_reached}")

    return best_path, goal_reached, best_cost

def calculate_total_distance(path, maze):
    total_distance = 0.0
    for position in path:
        x, y = position
        total_distance += maze[x][y]
    return total_distance

def visualize_maze(maze, path, start, goals, goal_reached):
    maze_display = np.zeros_like(maze, dtype=int)  # Inisialisasi dengan 0 (jalan)

    # Mengubah dinding (-1) menjadi 1 untuk visualisasi (hitam)
    maze_display[np.array(maze) == -1] = 1  # Dinding

    # Menandai jalur yang ditemukan
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
    # 0: putih (jalan), 1: hitam (dinding), 2: biru muda (jalur),
    # 3: merah (start), 4: hijau (tujuan dicapai), 5: kuning (tujuan lainnya)
    cmap = ListedColormap(['white', 'black', 'lightblue', 'red', 'green', 'yellow'])

    plt.figure(figsize=(8,8))
    plt.imshow(maze_display, cmap=cmap)
    plt.title('Visualisasi Maze dengan Jalur Terbaik (ACO)')
    plt.axis('off')
    plt.show()

def main():
    # Membaca maze dari file teks
    maze = load_maze_from_file("Rute/maze.txt")

    # ================================================================>>>>>>>>>>>>
    start = (20, 12)  # Titik awal
    goals = [(0, 8), (3, 16)]  # Daftar titik tujuan

    # Memeriksa apakah start dan tujuan tidak di dinding
    if maze[start[0]][start[1]] == -1:
        print("Titik awal berada di dinding.")
        return
    for goal in goals:
        if maze[goal[0]][goal[1]] == -1:
            print(f"Titik tujuan {goal} berada di dinding.")
            return

    # Menjalankan algoritma Ant Colony Optimization
    best_path, goal_reached, best_cost = ant_colony_optimization(maze, start, goals)

    if best_path:
        total_distance = calculate_total_distance(best_path, maze)
        print("Jalur terbaik ditemukan dengan biaya:", best_cost)
        print("Total jarak jalur:", total_distance)
        print("Menuju titik tujuan:", goal_reached)
        print("Jalur:", best_path)
        visualize_maze(maze, best_path, start, goals, goal_reached)
    else:
        print("Tidak ada jalur yang ditemukan.")

if __name__ == "__main__":
    main()
