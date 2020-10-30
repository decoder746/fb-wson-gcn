from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

import networkx as nx
from sklearn.metrics import average_precision_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, edge_list, adj_list = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            # nclass=labels.max().item() + 1,
            nclass=128,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = list(map(lambda x:x.cuda(),adj))
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def loss(cos_sim,edge_list,tim,delta):
    # pos_edges_added = list(zip(*list(set(edge_list[tim]) - set(edge_list[tim-1]))))
    # neg_edges_added = list(zip(*list(set(edge_list[tim+1]) - set(edge_list[tim]))))
    # pos = cos_sim[pos_edges_added[0],pos_edges_added[1]]
    # neg = cos_sim[neg_edges_added[0],neg_edges_added[1]]
    pos = cos_sim[((adj_list[tim].to_dense() - adj_list[tim-1].to_dense())>0).bool()]
    neg = cos_sim[((adj_list[tim+1].to_dense() - adj_list[tim].to_dense())>0).bool()]
    n_1 = pos.shape[0]
    n_2 = neg.shape[0]
    pos = pos.unsqueeze(1).expand(n_1,n_2)
    neg = neg.unsqueeze(0).expand(n_1,n_2) + delta
    pos = neg - pos
    hinge = nn.ReLU()
    pos = hinge(pos)
    return torch.sum(pos)

def train(epoch,delta=0.5):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    loss_train = 0
    for tim in idx_train:
        output = model(features, adj[tim])
        out_norm = output/output.norm(dim=1)[:,None]
        cos_sim = torch.mm(out_norm,out_norm.transpose(0,1))
        loss_train += loss(cos_sim,edge_list,tim,delta)
    loss_train.backward()
    optimizer.step()

    loss_val = 0
    # if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
    model.eval()
    for tim in idx_val:
        output = model(features, adj[tim])
        out_norm = output/output.norm(dim=1)[:,None]
        cos_sim = torch.mm(out_norm,out_norm.transpose(0,1))
        loss_val += loss(cos_sim,adj,tim,delta)
    if epoch%5 == 0:
        output = model(features,adj[80])
        out_norm = output/output.norm(dim=1)[:,None]
        cos_sim = torch.mm(out_norm,out_norm.transpose(0,1))
        edges_added = list(set(edge_list[99]) - set(edge_list[80]))
        G = nx.empty_graph()
        G.add_nodes_from(range(3000))
        G.add_edges_from(edge_list[80])
        non_edges = list(set(nx.non_edges(G)) - set(edges_added))
        scores = cos_sim[list(zip(*edges_added))[0],list(zip(*edges_added))[1]].tolist() + \
                cos_sim[list(zip(*non_edges))[0],list(zip(*non_edges))[1]].tolist()
        ground_truth = [1]*len(edges_added) + [0]*len(non_edges)
        AP = average_precision_score(ground_truth,scores)
        print(f'Epoch: {epoch} AP: {AP}')

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
        #   'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
        #   'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
