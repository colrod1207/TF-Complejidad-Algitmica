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