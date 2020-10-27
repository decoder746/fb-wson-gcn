import sys,os

file_name = "fb3k.feat"
num_nodes = 3000
with open(file_name,'w') as f:
    for i in range(num_nodes):
        feat = ["0"]*num_nodes
        feat[i] = "1"
        st = str(i) + " "+ " ".join(feat) + " 0\n"
        f.write(st)