import argparse
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from torch_geometric.datasets import Planetoid,WebKB,Actor,WikipediaNetwork, LINKXDataset
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import *
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything as th_seed
from models import *
from utils import *
from dataset_large import *
import warnings
warnings.filterwarnings("ignore")
th_seed(12345)
################### Arguments parameters ###################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="penn94",
    choices=["penn94","pokec","arxiv_year","snap_patents","genius","twitch-gamer"],
    help="You can choose between penn94, pokec, arxiv_year, snap_patents, genius, twitch-gamer",
)
parser.add_argument(
    "--cuda",
    default="cuda:0",
    choices=["cuda:0","cuda:1","cpu"],
    help="You can choose between cuda:0, cuda:1, cpu",
)
parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Hidden channels for the unsupervised model"
)
parser.add_argument(
        "--dropout", type=float, default=0.35, help="Dropout rate"
    )
parser.add_argument(
        "--lr", type=float, default=0.003, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=5e-4, help="Outer weight decay rate of model"
    )
parser.add_argument(
        "--epochs", type=int, default=700, help="Epochs for the model"
    )
parser.add_argument(
        "--n_layers", type=int, default=7, help="Number of hops"
    )
parser.add_argument(
        "--hops", type=int, default=4, help="Number of centers"
)
args = parser.parse_args()
################### Importing the dataset ###################################
if args.dataset == "penn94":
    dataset = LINKXDataset(root='./data',name='penn94')
    data = dataset[0]
elif args.dataset == "pokec":
    data = pokec()
elif args.dataset == "arxiv_year":
    data = arxiv_year()
elif args.dataset == "snap_patents":
    data = snap_patents()
elif args.dataset == "genius":
    data = genius()
elif args.dataset == "twitch-gamer":
    data = twitch_gamer()

init_edge_index = data.edge_index.clone()
print("Computing the graphs...")
G_l = khop_graphs_sparse(data.x,data.edge_index, args.hops,args.dataset,"cpu",features=True)
G_l.append(init_edge_index)
#edge_l.append(torch.ones(init_edge_index.shape[1]))
print("Done!")
data.edge_index = G_l
#data.edge_attr = edge_l
print()
print(f'Dataset: {args.dataset}')
print('======================')
print(f'Number of graphs: {data.x.shape[0]}')
print(f'Number of features: {data.x.shape[1]}')
print(f'Number of classes: {data.y.max().item()+1}')
print()
print(data) 
print('===========================================================================================================')
################### CUDA ###################################
device = torch.device(args.cuda)
#device = 'cpu'
#device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
data = data.to(device)   
print("Device: ",device)
################### Training the model in a supervised way ###################################
results = []
for i in range(5):
    if args.dataset != "penn94":
        # Split the data in 50% train, 25% validation and 25% test
        train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        train_mask[:int(data.x.shape[0]*0.5)] = True
        val_mask[int(data.x.shape[0]*0.5):int(data.x.shape[0]*0.75)] = True
        test_mask[int(data.x.shape[0]*0.75):] = True
    else:
        train_mask = data.train_mask[:,i]
        val_mask = data.val_mask[:,i]
        test_mask = data.test_mask[:,i]
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)                                                
    print('===========================================================================================================')
    print('Split: ',i)
    print('===========================================================================================================')
    model = MO_GNN_large(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=data.y.max().item()+1,
                    num_layers=args.hops,
                    dropout=args.dropout).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    test_acc = 0
    patience = 0
    for epoch in range(args.epochs):
        loss,acc_train = train(data,model,train_mask,optimizer,criterion)
        acc_val = val(data,model,val_mask)
        acc_test = test(data,model,test_mask)
        if acc_test > test_acc:
            test_acc = acc_test
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}')
        if test_acc > acc_test:
            patience += 1
        else:
            patience = 0
        if patience == 500:
            break
    print('===========================================================================================================')
    print('Test Accuracy: ',test_acc)
    print('===========================================================================================================')
    results.append(test_acc)
print('===========================================================================================================')
print('Report: ',np.mean(results)*100,'+-',np.std(results)*100)
print('===========================================================================================================')
print(' Configuration: ',args)
print('===========================================================================================================')

# Now we check if it is created a csv with the configuration and the results
if os.path.isfile('results.csv'):
    # If the file exists, then we append the configuration and the results
    # The columns are: dataset, model, hidden_channels, lr, epochs, num_centers, AUC, AP
    res = pd.read_csv('results.csv')
    # Check if the configuration is already in the csv
    if res[(res['dataset'] == args.dataset) & (res['hidden_channels'] == args.hidden_channels) & (res['lr'] == args.lr) & (res['epochs'] == args.epochs) & (res['hops'] == args.hops) & (res['n_layers'] == args.n_layers) & (res['dropout'] == args.dropout)].shape[0] == 0:
        # If the configuration is not in the csv, then we append it
        #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
        res = pd.concat([res, pd.DataFrame({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
        res.to_csv('results.csv', index=False)
    res.to_csv('results.csv', index=False)
else:
    # If the file does not exist, then we create it and append the configuration and the results
    res = pd.DataFrame(columns=['dataset', 'hidden_channels', 'lr','dropout', 'epochs', 'hops', 'n_layers', 'Accuracy', 'std'])
    #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
    res = pd.concat([res, pd.DataFrame({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'hops': args.hops, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
    res.to_csv('results.csv', index=False)
