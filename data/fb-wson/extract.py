import networkx as nx
import random
from random import sample
import copy
import math
import numpy as np

from collections import Counter


random.seed(0)
filename = "../dblk/DBLP10k.csv"
# filename = "fb-wosn-friends.edges"
# community = "fb3k.csv"
# wotime = "fb_correct.csv"
wotime = "../dblk/edge_list.csv"
# ran_comm = "fb5k_1.csv"
# ran_comm2 = "fb5k_2.csv"

author_names = []

with open(filename,'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.rstrip() == "":
            break
        line = line.rstrip().split(";")
        author_names.append(line[19])
        author_names.append(line[-1])
        print(line[19],line[-1])

print(len(author_names),len(set(author_names)))

# result = [item for item, c in Counter(author_names).most_common()] 
# node_list = result[:3000]

# with open(filename,'r') as f:
#     with open(community,'w') as c:
#         lines = f.readlines()
#         edge_count = 0
#         for line in lines:
#             if line.rstrip() == "":
#                 break
#             line = line.rstrip().split(" ")
#             line = list(map(int,line))
#             if line[3]!=0 and line[0] in node_list and line[1] in node_list:
#                 c.write(f'{node_list.index(line[0])} {node_list.index(line[1])} {line[3]}\n')
dist_nodes = list(set(author_names))

with open(filename,'r') as f:
    with open(wotime,'w') as g:
        lines = f.readlines()
        for line in lines:
            if line.rstrip() == "":
                break
            line = line.rstrip().split(";")
            # line = list(map(int,line))
            g.write(f'{dist_nodes.index(line[19])} {dist_nodes.index(line[-1])} {max(line[12],line[-8])}\n')


# comm1 = random.sample(dist_nodes,5000)
# comm2 = random.sample(list(set(author_names)-set(comm1)),5000)

# with open(wotime,'r') as f:
#     with open(ran_comm,'w') as c:
#         lines = f.readlines()
#         edge_count = 0
#         for line in lines:
#             if line.rstrip() == "":
#                 break
#             line = line.rstrip().split(" ")
#             line = list(map(int,line))
#             if line[0] in comm1 and line[1] in comm1 and line[2]!=0:
#                 c.write(f'{comm1.index(line[0])} {comm1.index(line[1])} {line[2]}\n')

# with open(wotime,'r') as f:
#     with open(ran_comm2,'w') as c:
#         lines = f.readlines()
#         edge_count = 0
#         for line in lines:
#             if line.rstrip() == "":
#                 break
#             line = line.rstrip().split(" ")
#             line = list(map(int,line))
#             if line[0] in comm2 and line[1] in comm2 and line[2]!=0:
#                 c.write(f'{comm2.index(line[0])} {comm2.index(line[1])} {line[2]}\n')