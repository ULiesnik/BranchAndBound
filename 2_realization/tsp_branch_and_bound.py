import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

class Node:
    def __init__(self, level, path, reduced_matrix, cost, path_cost):
        self.level = level
        self.path = path
        self.reduced_matrix = reduced_matrix
        self.cost = cost         
        self.path_cost = path_cost 

    def __lt__(self, other):
        return self.cost < other.cost

def reduce_matrix(matrix):
    """Зменшує матрицю, обчислює вартість зменшення."""
    reduced_matrix = matrix.copy()
    row_min = np.min(reduced_matrix, axis=1)
    row_min[row_min == np.inf] = 0
    reduced_matrix -= row_min[:, np.newaxis]
    
    col_min = np.min(reduced_matrix, axis=0)
    col_min[col_min == np.inf] = 0
    reduced_matrix -= col_min
    
    cost = np.sum(row_min) + np.sum(col_min)
    return reduced_matrix, cost

def calculate_cost(matrix, path, prev_cost, path_cost, distance_matrix):
    """Оновлює зменшену матрицю і вартість на основі поточного шляху."""
    last = path[-2]
    current = path[-1]
    
    reduced_matrix = matrix.copy()
    
    reduced_matrix[last, :] = np.inf
    reduced_matrix[:, current] = np.inf
    reduced_matrix[current, 0] = np.inf 
    
    reduced_matrix, reduction_cost = reduce_matrix(reduced_matrix)
    new_cost = prev_cost + distance_matrix[last][current] + reduction_cost
    new_path_cost = path_cost + distance_matrix[last][current]
    
    return reduced_matrix, new_cost, new_path_cost

def format_matrix(matrix):
    return pd.DataFrame(matrix).style.format(precision=2).hide(axis='index')

def tsp_branch_and_bound(distance_matrix):
    """Розв'язує задачу комівояжера методом гілок та меж."""
    matrix = np.array(distance_matrix)
    pq = []
    reduced_matrix, cost = reduce_matrix(matrix)
    root = Node(0, [0], reduced_matrix, cost, 0)
    heapq.heappush(pq, root)
    
    best_cost = np.inf
    best_path = []
    
    results = []
    while pq:
        node = heapq.heappop(pq)

        # print(f"Крок {node.level}")
        # print(f"Шлях {node.path}")
        # print(f"Поточна матриця {node.reduced_matrix}")
        # print(f"Поточна вартість {node.cost}")
        
        if node.level == len(matrix) - 1:
            last = node.path[-1]
            return_cost = distance_matrix[last][0]
            if return_cost == np.inf:
                continue 
            current_path_cost = node.path_cost + return_cost
            if current_path_cost < best_cost:
                best_cost = current_path_cost
                best_path = node.path + [0]
            continue

        results.append({
            "Level": node.level,
            "Path": " > ".join(map(str,node.path)),
            "Cost": node.cost,
            "Path cost": node.path_cost,
            "Reduced Matrix": node.reduced_matrix.copy()
        })
        
        for i in range(1, len(matrix)):
            if i not in node.path:
                new_path = node.path + [i]
                reduced_matrix_new, new_cost, new_path_cost = calculate_cost(
                    node.reduced_matrix, new_path, node.cost, node.path_cost, distance_matrix
                )
                
                if new_cost < best_cost:
                    new_node = Node(
                        node.level + 1,
                        new_path,
                        reduced_matrix_new,
                        new_cost,
                        new_path_cost
                    )
                    heapq.heappush(pq, new_node)
                    
    return best_cost, best_path, results

def draw_graph(matrix, path):
    G = nx.DiGraph()
    num_nodes = len(matrix)
    pos = {}
    radius = 10
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))
        G.add_node(i, pos=pos[i])
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix[i][j] != np.inf:
                G.add_edge(i, j, weight=matrix[i][j])
    
    labels = nx.get_edge_attributes(G, 'weight')
    
    plt.figure(figsize=(10, 10))
    
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', width=1.0, arrows=True, alpha=0.5)
    
    edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    for u, v in edges_in_path:
        if matrix[u][v] != np.inf:
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=[(u, v)], 
                edge_color='blue', 
                width=2.5, 
                arrows=True, 
                arrowstyle='-|>', 
                arrowsize=20, 
                alpha=0.7
            )
    
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color="red", font_size=10)
    
    path_labels = {edge: labels[edge] for edge in edges_in_path if edge in labels}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=path_labels, font_color="blue", font_size=12)
    
    plt.title("Задача комівояжера: граф з відображенням шляху та ваги")
    plt.axis('off')
    
    return(plt)

distance_matrix = [
    [np.inf, 20, 30, 10, 11],
    [15, np.inf, 16, 4, 2],
    [3, 5, np.inf, 2, 4],
    [19, 6, 18, np.inf, 3],
    [16, 4, 7, 16, np.inf]
]

best_cost, best_path, results = tsp_branch_and_bound(distance_matrix)

print(f"Найменша вартість: {best_cost}")
print(f"Найкращий шлях: {best_path}")

for result in results:
    print(f"# Рівень {result['Level']}")
    print(f"Шлях {result['Path']}")
    print(f"Вартість шляху {result['Path cost']}")
    print("Знижена матриця")
    df = pd.DataFrame(result["Reduced Matrix"])
    print(df)
    print("")