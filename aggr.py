import networkx as nx
import random
from random import sample
import copy
import math
import numpy as np
from sklearn.metrics import average_precision_score

# filename = "data/fb-wson/fb3k.csv"
filename = "data/fb-wson/fb_correct.csv"
random.seed(0)
edge_list = []
test_fraction = 0.2


def create_scores_common_neigh(graph,edge_list=None):
    for u,v in edge_list:
        value = len(list(nx.common_neighbors(graph,u,v)))
        yield (u,v,value)


num_nodes = 0
with open(filename,'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.rstrip() == "":
            break
        line = list(map(int,line.rstrip().split(" ")))
        num_nodes = max(num_nodes,line[0],line[1])
        edge_list.append((line[0],line[1],line[2]))
    
print(f'Num nodes: {num_nodes} Num edges: {len(edge_list)}')
edge_list = sorted(edge_list,key=lambda x: x[2])
train_len = int((1-test_fraction)*len(edge_list))
train_edge_list = list(map(lambda x:(x[0],x[1]),edge_list[:train_len]))
test_edge_list = list(map(lambda x:(x[0],x[1]),edge_list[train_len:]))
g = nx.Graph()
g.add_nodes_from(range(num_nodes+1))
g.add_edges_from(train_edge_list)
non_edge_list = list(set(list(nx.non_edges(g))) - set(test_edge_list))
test_non_edge_list = sample(non_edge_list,len(non_edge_list)*test_fraction)
ground_truth = [1]*len(test_edge_list)+[0]*len(test_non_edge_list)

print("Created dataset")

adam = list(nx.adamic_adar_index(g,test_edge_list+test_non_edge_list))
adam_map = average_precision_score(ground_truth,list(map(lambda x:x[2],adam)))
print(f'Adamic ap is {adam_map}')

adam = list(nx.jaccard_coefficient(g,test_edge_list+test_non_edge_list))
adam_map = average_precision_score(ground_truth,list(map(lambda x:x[2],adam)))
print(f'Jacard ap is {adam_map}')

adam = list(create_scores_common_neigh(g,test_edge_list+test_non_edge_list))
adam_map = average_precision_score(ground_truth,list(map(lambda x:x[2],adam)))
print(f'CN ap is {adam_map}')

