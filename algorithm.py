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