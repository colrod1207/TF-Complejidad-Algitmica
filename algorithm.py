import os
import pandas as pd
from collections import defaultdict
import heapq as hq
import math
import graphviz as gv
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

dataset = 'dataset.csv'
df = pd.read_csv(dataset)

def create_adj_list(df):
    start_time = time.time()
    G = defaultdict(list)
    for _, row in df.iterrows():
        start_node = int(row['Start_Node_ID'])
        end_node = int(row['End_Node_ID'])
        operation_cost = row['Operation_Cost']
        G[start_node].append((end_node, operation_cost))
        G[end_node].append((start_node, operation_cost))
    print(f"Lista de adyacencia creada en {time.time() - start_time:.2f} segundos.")
    return G

def dijkstra(G, s):
    start_time = time.time()
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

    print(f"Dijkstra ejecutado en {time.time() - start_time:.2f} segundos.")
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

def drawG_al(G, path_nodes):
    start_time = time.time()

    graph = gv.Graph("grafo")
    graph.graph_attr["layout"] = "dot"
    graph.edge_attr["color"] = "gray"
    graph.node_attr["color"] = "orangered"
    graph.node_attr["width"] = "0.1"
    graph.node_attr["height"] = "0.1"
    graph.node_attr["fontsize"] = "8"
    graph.node_attr["fontcolor"] = "mediumslateblue"
    graph.node_attr["fontname"] = "monospace"
    graph.edge_attr["fontsize"] = "8"
    graph.edge_attr["fontname"] = "monospace"

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        weight = next(w for n, w in G[u] if n == v)
        graph.node(str(u))
        graph.node(str(v))
        graph.edge(str(u), str(v), label=str(weight), color="orange", penwidth="2")

    print(f"Grafo dibujado en {time.time() - start_time:.2f} segundos.")
    return graph

def main():
    G = create_adj_list(df)
    start_node = get_valid_node("Ingrese el nodo de inicio: ", G)
    end_node = get_valid_node("Ingrese el nodo de destino: ", G)

    shortest_path, min_cost, steps = shortest_path_between_nodes(G, start_node, end_node)
    print(f"El camino más corto desde {start_node} hasta {end_node} es: {steps} con un costo operativo mínimo de: {min_cost}")

    graph = drawG_al(G, shortest_path)

    output_file = 'grafo'
    graph.render(output_file, format='png', view=False)
    print(f"Gráfico renderizado y guardado en {output_file}.png")

    img = mpimg.imread(f'{output_file}.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
