import networkx as nx
import random
from random import sample
import copy
import math
import numpy as np

filename = "fb3k.csv"
random.seed(0)
nbr_dict = {}

with open(filename,'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.rstrip() == "":
            break
        line = list(map(int,line.rstrip().split(" ")))
        if line[0] in nbr_dict.keys():
            nbr_dict[line[0]].append((line[1],line[2]))
        else:
            nbr_dict[line[0]] = []
            nbr_dict[line[0]].append((line[1],line[2]))
        if line[1] in nbr_dict.keys():
            nbr_dict[line[1]].append((line[0],line[2]))
        else:
            nbr_dict[line[1]] = []
            nbr_dict[line[1]].append((line[0],line[2]))


num_nodes = len(nbr_dict.keys())
thre = 1000

def split_train_test(test_fraction):
    train_nbr,train_non_nbr, test_nbr, test_non_nbr = {},{},{},{}
    test_edge_list = []
    print(f'Number of nodes {len(nbr_dict.keys())}')
    count = 0
    for key in nbr_dict.keys():
        count+=1
        if count%1000 == 0:
            print(f'{count} nodes completed')
        sort_list = list(map(lambda x:x[0],sorted(nbr_dict[key],key=lambda x:x[1])))
        train_nbr_len = int((1-test_fraction)*len(sort_list))
        test_non_nbr_len = int(test_fraction*(num_nodes-len(sort_list)))
        train_nbr[key] = sort_list[:train_nbr_len]
        test_nbr[key] = sort_list[train_nbr_len:]
        # train_non_nbr[key] = sample(list(set(range(num_nodes))-set(sort_list)),train_non_nbr_len)
        # test_non_nbr[key] = list(set(range(num_nodes))- set(train_non_nbr))
        test_edge_list.extend([(key,b) for b in test_nbr[key]])
        test_edge_list.extend([(key,b) for b in sample(list(set(range(num_nodes)) - set(sort_list)),test_non_nbr_len)])
    return train_nbr,test_nbr,test_edge_list


train_nbr,test_nbr,test_all = split_train_test(0.2)
print("Dictionaries created")

def adamic_adar(train_nbr,edge_list):
    for u,v in edge_list:
        try:
            yield (u,v,sum(1 / math.log(len(train_nbr[w])) if len(train_nbr[w]) > 1 else 0 for w in set(train_nbr[u]).intersection(set(train_nbr[v]))))
        except ZeroDivisionError:
            yield (u,v,0)

def create_scores_common_neigh(train_nbr,edge_list=None):
    for u,v in edge_list:
        value = len(set(train_nbr[u]).intersection(train_nbr[v]))
        yield (u,v,value)

def computePrecisionCurve(predicted_edge_list, dict_nbr, max_k=-1):
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))
    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)
    precision_scores = []
    delta_factors = []
    correct_edge = 0
    first_time = 1
    rr = 0
    for i in range(max_k):
        if sorted_edges[i][1] in dict_nbr :
            if first_time:
                rr =  1.0/(i+1)
                first_time = 0
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors,rr


def computeMAP_MRR(predicted_edge_list, graph, max_k=-1):
    node_num = num_nodes
    node_edges = []
    for i in range(node_num):
        node_edges.append([])
    for (st, ed, w) in predicted_edge_list:
        node_edges[st].append((st, ed, w))
    print("Completed iterating over the generator of scores")
    node_AP = [0.0] * node_num
    count = 0
    mrr = 0
    for i in range(node_num):
        if len(test_nbr[i]) == 0:
            continue
        count += 1
        precision_scores, delta_factors,rr = computePrecisionCurve(node_edges[i], test_nbr[i], max_k)
        mrr += rr
        precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
        if (sum(delta_factors) == 0):
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count , mrr/count

adamic = adamic_adar(train_nbr,test_all)
adamic_map,adamic_mrr = computeMAP_MRR(adamic,test_nbr)
print("Adamic adar")
print("MAP={}".format(adamic_map))
print("MRR={}".format(adamic_mrr))

cn = create_scores_common_neigh(train_nbr,test_all)
cn_map,cn_mrr = computeMAP_MRR(cn,test_nbr)
print("Common Neighbor")
print("MAP={}".format(cn_map))
print("MRR={}".format(cn_mrr))