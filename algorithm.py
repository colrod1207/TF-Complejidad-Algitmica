import pandas as pd
from collections import defaultdict
import heapq as hq
import math

dataset = 'dataset.csv'
df = pd.read_csv(dataset)

def create_adj_list(df):
    G = defaultdict(list)
    for _, row in df.iterrows():
        start_node = int(row['Start_Node_ID'])
        end_node = int(row['End_Node_ID'])
        operation_cost = row['Operation_Cost']
        G[start_node].append((end_node, operation_cost))
        G[end_node].append((start_node, operation_cost)) 
    return G

G = create_adj_list(df)

for key, value in G.items():
    print(f"{key}: {value}")

def dijkstra(G, s):
    n = len(G)
    visited = {k: False for k in G.keys()}
    path = {k: -1 for k in G.keys()}
    cost = {k: math.inf for k in G.keys()}
    
    cost[s] = 0
    pqueue = [(0, s)]
    while pqueue:
        g, u = hq.heappop(pqueue)
        if not visited[u]:
            visited[u] = True
            for v, w in G[u]:
                if not visited[v]:
                    f = g + w
                    if f < cost[v]:
                        cost[v] = f
                        path[v] = u
                        hq.heappush(pqueue, (f, v))
    
    return path, cost    

def shortest_path_between_nodes(G, start_node, end_node):
    path, cost = dijkstra(G, start_node)
    if cost[end_node] == math.inf:
        return f"No hay camino desde {start_node} hasta {end_node}.", None, None
    
    shortest_path = []
    current_node = end_node
    while current_node != -1:
        shortest_path.append(current_node)
        current_node = path[current_node]
    
    shortest_path.reverse()
    
    steps_with_weights = []
    for i in range(len(shortest_path) - 1):
        u = shortest_path[i]
        v = shortest_path[i + 1]
        weight = next(w for n, w in G[u] if n == v)
        steps_with_weights.append(f"{u} --({weight})--> {v}")
    
    steps = " -> ".join(steps_with_weights)
    
    return shortest_path, cost[end_node], steps

G = create_adj_list(df)

def get_valid_node(prompt, G):
    while True:
        try:
            node = int(input(prompt))
            if node in G:
                return node
            else:
                print(f"El nodo {node} no existe en el grafo. Inténtelo de nuevo.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número entero.")
