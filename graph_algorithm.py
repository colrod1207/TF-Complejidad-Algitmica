import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import os


def create_graph():
    
    complete_dataset = pd.read_csv('dataset.csv', delimiter=',')

    G = nx.DiGraph()

    for _, row in complete_dataset.iterrows():
        G.add_edge(row['Start_Node_ID'], row['End_Node_ID'], length=row['Pipe_Length'], capacity=row['Pipe_Capacity'], cost=row['Operation_Cost'])

    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42, k=0.1)  

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue')

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=10)

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title('Grafo del dataset de tuberias')

    img_path = 'static/graph.png'
    plt.savefig(img_path)
    plt.close()
    return img_path
