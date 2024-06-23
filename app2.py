import os
import pandas as pd
from collections import defaultdict
import heapq as hq
import math
import graphviz as gv
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx

# Ajusta esta ruta si es necesario
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

dataset = 'dataset.csv'
df = pd.read_csv(dataset)


def create_adj_list():
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

    output_file = 'grafo'
    graph.render(output_file, format='png', view=False)
    print(f"Grafo dibujado y guardado en {output_file}.png")
    return f'{output_file}.png'


def create_graph():
    complete_dataset = pd.read_csv('dataset.csv', delimiter=',')

    G = nx.DiGraph()

    for _, row in complete_dataset.iterrows():
        G.add_edge(row['Start_Node_ID'], row['End_Node_ID'], length=row['Pipe_Length'], capacity=row['Pipe_Capacity'],
                   cost=row['Operation_Cost'])

    plt.figure(figsize=(30, 30))  # Ajusta el tamaño de la figura para que sea más grande
    pos = nx.spring_layout(G, seed=42, k=0.1)

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title('Grafo del dataset de tuberias')

    img_path = 'static/graph.png'
    plt.savefig(img_path)
    plt.close()
    return img_path


def show_full_graph():
    graph_path = create_graph()
    img = mpimg.imread(graph_path)
    plt.figure(figsize=(10, 10))  # Ajusta el tamaño de la ventana que muestra el grafo completo
    plt.imshow(img)
    plt.axis('off')
    canvas.get_tk_widget().pack_forget()  # Elimina el canvas anterior
    new_canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    new_canvas.draw()
    new_canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)
    plt.close()


def calculate_and_show_path():
    start_node = int(start_node_entry.get())
    end_node = int(end_node_entry.get())
    G = create_adj_list()
    shortest_path, min_cost, steps = shortest_path_between_nodes(G, start_node, end_node)

    if shortest_path:
        graph_path = drawG_al(G, shortest_path)
        img = mpimg.imread(graph_path)
        plt.figure(figsize=(5, 4))  # Mantiene el tamaño original de la figura para el camino más corto
        plt.imshow(img)
        plt.axis('off')
        canvas.get_tk_widget().pack_forget()  # Elimina el canvas anterior
        new_canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
        new_canvas.draw()
        new_canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)
        plt.close()
        result_label.config(
            text=f"El camino más corto desde {start_node} hasta {end_node} es: {steps} con un costo operativo mínimo de: {min_cost}")
    else:
        messagebox.showerror("Error", "No hay camino entre los nodos especificados")


# Crear la ventana principal
root = tk.Tk()
root.title("Trabajo Final de Algoritmos")

# Crear el frame principal
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Titulo
title_label = ttk.Label(frame, text="Trabajo Final de Algoritmos", font=("Helvetica", 16))
title_label.grid(row=0, column=0, columnspan=2)

# Entrada para el nodo de inicio
start_node_label = ttk.Label(frame, text="Nodo de Inicio:")
start_node_label.grid(row=1, column=0, sticky=tk.W)
start_node_entry = ttk.Entry(frame, width=10)
start_node_entry.grid(row=1, column=1, sticky=tk.W)

# Entrada para el nodo de fin
end_node_label = ttk.Label(frame, text="Nodo de Fin:")
end_node_label.grid(row=2, column=0, sticky=tk.W)
end_node_entry = ttk.Entry(frame, width=10)
end_node_entry.grid(row=2, column=1, sticky=tk.W)

# Botón para calcular el camino más corto
calculate_button = ttk.Button(frame, text="Calcular Camino Más Corto", command=calculate_and_show_path)
calculate_button.grid(row=3, column=0, columnspan=2)

# Botón para mostrar el grafo completo
show_graph_button = ttk.Button(frame, text="Mostrar Grafo Completo", command=show_full_graph)
show_graph_button.grid(row=4, column=0, columnspan=2)

# Etiqueta para mostrar el resultado
result_label = ttk.Label(frame, text="", wraplength=400)
result_label.grid(row=5, column=0, columnspan=2)

# Figura de Matplotlib para mostrar el grafo
fig = plt.figure(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)

# Iniciar el bucle principal de la interfaz
root.mainloop()
