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
from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx

app = Flask(__name__)

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

dataset = 'dataset2.0.csv'
df = pd.read_csv(dataset)


def create_adj_list():
    start_time = time.time()
    G = defaultdict(list)
    for _, row in df.iterrows():
        start_node = int(row['ID_Nodo_Inicio'])
        end_node = int(row['ID_Nodo_Final'])
        operation_cost = row['Costo_de_Operacion']
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
    print( "antes ",shortest_path)
    shortest_path.reverse()
    print("despues ",shortest_path)
    steps_with_weights = []
    for i in range(len(shortest_path) - 1):
        u = shortest_path[i]
        v = shortest_path[i + 1]
        weight = min(w for n, w in G[u] if n == v)
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

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        weight = min(w for n, w in G[u] if n == v)
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
    for index, row in df.iterrows():
        try:
            G.add_edge(row['ID_Nodo_Inicio'], row['ID_Nodo_Final'],
                length=row['Longitud_(km)'],
                capacity=row['Flujo_Necesario_(m3)'],
                energy_cost=row['Costo_de_energia'],
                labor_cost=row['Mano_de_obra'],
                material=row['Material'],
                operation_cost=row['Costo_de_Operacion'])
        except KeyError as e:
            print(f"Error al acceder a la columna: {e}")

    plt.figure(figsize=(30, 30))
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
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    canvas.get_tk_widget().pack_forget()
    new_canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    new_canvas.draw()
    new_canvas.get_tk_widget().grid(row=6, column=0, columnspan=2,sticky=(tk.W, tk.E, tk.N, tk.S))
    plt.close()


def calculate_and_show_path():
    start_node = int(start_node_entry.get())
    end_node = int(end_node_entry.get())
    G = create_adj_list()
    shortest_path, min_cost, steps = shortest_path_between_nodes(G, start_node, end_node)

    if shortest_path:
        graph_path = drawG_al(G, shortest_path)
        img = mpimg.imread(graph_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        canvas.get_tk_widget().pack_forget()
        new_canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
        new_canvas.draw()
        new_canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)
        plt.close()
        #result_label.config(
            #text=f"El camino más corto desde {start_node} hasta {end_node} es:\n {steps} \ncon un costo operativo mínimo de: {min_cost}")
        # Limpiar la tabla antes de insertar nuevos datos
        for row in tree.get_children():
            tree.delete(row)

        # Insertar datos en el Treeview
        for i in range(len(shortest_path) - 1):
            u = shortest_path[i]
            v = shortest_path[i + 1]
            weight = min(w for n, w in G[u] if n == v)
            tree.insert("", tk.END, values=(u, v, weight))
        tree.insert("", tk.END, values=(" ", "Total:", min_cost))
    else:
        messagebox.showerror("Error", "No hay camino entre los nodos especificados")


root = tk.Tk()
root.title("Trabajo Final de Algoritmos")
root.geometry("1600x900")
root.configure(bg="#fdad1b")

# Crear y configurar estilo
style = ttk.Style()
style.configure('TFrame', background='#fdad1b')
style.configure('TLabel', background='#fdad1b', foreground='black')
style.configure('TButton', font=("Helvetica", 16), foreground='black')
style.map('TButton', background=[('!active', '#ff9800'), ('active', '#ff9800')])

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)
frame.grid_columnconfigure(2, weight=1)
frame.grid_rowconfigure(6, weight=1)


title_label = ttk.Label(frame, text="Trabajo Final de Algoritmos", font=("Helvetica", 25, "bold"))
title_label.grid(row=0, column=1, pady=(0, 10))

start_node_label = ttk.Label(frame, text="Nodo de Inicio:", font=("Helvetica", 16, "bold"))
start_node_label.grid(row=1, column=0, sticky=tk.E, pady=(5, 5))
start_node_entry = ttk.Entry(frame, width=10, font=("Helvetica", 12))
start_node_entry.grid(row=1, column=1, pady=(0, 5))

end_node_label = ttk.Label(frame, text="Nodo de Fin:", font=("Helvetica", 16, "bold"))
end_node_label.grid(row=2, column=0, sticky=tk.E, pady=(5, 5))
end_node_entry = ttk.Entry(frame, width=10, font=("Helvetica", 12))
end_node_entry.grid(row=2, column=1, columnspan=1 , pady=(0, 5))

calculate_button = ttk.Button(frame, text="Calcular Camino Más Corto", command=calculate_and_show_path)
calculate_button.grid(row=3, column=0, pady=(0, 0))

show_graph_button = ttk.Button(frame, text="Mostrar Grafo Completo", command=show_full_graph)
show_graph_button.grid(row=3, column=1, pady=(0, 0))

end_node_label = ttk.Label(frame, text="Reporte", font=("Helvetica", 16, "bold"))
end_node_label.grid(row=3, column=2, pady=(0, 0))

result_label = ttk.Label(frame, text="", wraplength=800, font=("Helvetica", 15))
result_label.grid(row=4, column=1, pady=(0, 10))

# Crear el Treeview
columns = ("Nodo de Inicio", "Nodo de Fin", "Costo")
tree = ttk.Treeview(frame, columns=columns, show='headings')

# Configurar encabezados de columnas
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, minwidth=0, width=130)

# Insertar datos de ejemplo
tree.insert("", tk.END, values=(" ", "Total", "--"))

tree.grid(row=6, column=2, pady=(0, 0), sticky=( tk.N, tk.S))

fig = plt.figure(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S),padx=(60,0))

root.mainloop()
