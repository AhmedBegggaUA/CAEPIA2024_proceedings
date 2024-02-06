import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np
import psutil
import os   
# Usamos pickle para guardar los datos
import pickle
def khop_graphs(x,edge_index, k):
    hops = []
    # Get the number of nodes
    N = edge_index.max().item() + 1
    # Create the adjacency matrix
    A = torch.zeros(N,N)
    A[edge_index[0],edge_index[1]] = 1
    # Add self loops
    A = A + torch.eye(N)
    # Degree matrix
    degrees = torch.sum(A,dim=1)
    D = torch.diag(degrees)
    D_tilde = torch.pow(D,-0.5)
    D_tilde[D_tilde == float('inf')] = 0
    A_tilde = torch.matmul(torch.matmul(D_tilde,A),D_tilde)
    # Compute A_tilde^k
    A_tilde_k = A_tilde
    hops.append(A_tilde_k.clone())
    for i in range(k-1):
        A_tilde_k = torch.sparse.mm(A_tilde_k,A_tilde)
        hops.append(A_tilde_k.clone())
    return hops


def khop_graphs_sparse(x, edge_index, k,name,device,features=True):
    # Checkas si ya existe el pickle
    if os.path.exists('hops_'+name+'_k_'+str(k)+'.pkl'):
        with open('hops_'+name+'_k_'+str(k)+'.pkl', 'rb') as f:
            hops = pickle.load(f)
        print("Loaded hops from file")
        return hops
    hops = list()
    N = edge_index.max().item() + 1
    # Create the adjacency matrix
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (N, N)).to(device)
    # Add self loops
    I = torch.sparse_coo_tensor(torch.arange(N).unsqueeze(0).repeat(2, 1), torch.ones(N), (N, N)).to(device)
    A = A + I
    # Degree matrix
    degrees = torch.sparse.sum(A, dim=1)
    degrees = torch.pow(degrees, -0.5)
    # Get the indices of the diagonal elements
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1).to(device)
    values = degrees.coalesce().values().to(device)
    # Create the sparse diagonal matrix
    D_tilde = torch.sparse_coo_tensor(indices, values, (N, N)).to(device)
    A_tilde = torch.sparse.mm(torch.sparse.mm(D_tilde, A), D_tilde)
    # Compute A_tilde^k
    A_tilde_k = A_tilde.clone().to(device)
    hops.append(A_tilde_k.clone())
    for i in range(k - 1):
        print("Computing k: ", i+1, " of ", k-1)
        if device == 'cpu':
            # Mostramos cuanta memoria ram del sistema estamos usando
            print("Ram memory: ", psutil.virtual_memory().percent, "%")
        else:
            # Mostramos cuanta memoria estamos usando
            print(torch.cuda.memory_allocated(device=device), "out of ", torch.cuda.max_memory_allocated(device=device))
        A_tilde_k = torch.sparse.mm(A_tilde_k, A_tilde)
        hops.append(A_tilde_k.clone())
    # Save the hops
    with open('hops_'+name+'_k_'+str(k)+'.pkl', 'wb') as f:
        pickle.dump(hops, f)
    return hops        
        
    


def train(data,model,train_mask,optimizer,criterion):
    model.train()
    optimizer.zero_grad()
    # Get the output of the model
    out = model(data.x, data.edge_index, data.edge_attr)
    # Compute the loss
    loss = criterion(out[train_mask], data.y[train_mask])
    # Compute the accuracy
    pred = out.argmax(dim=1)
    train_correct = pred[train_mask] == data.y[train_mask]
    acc = int(train_correct.sum()) / int(train_mask.sum())
    # Backpropagation
    loss.backward()
    optimizer.step()
    return loss, acc

def val(data,model,val_mask):
    model.eval()
    # Get the output of the model
    out = model(data.x, data.edge_index, data.edge_attr)
    # Compute the accuracy
    pred = out.argmax(dim=1)
    val_correct = pred[val_mask] == data.y[val_mask]
    acc = int(val_correct.sum()) / int(val_mask.sum())
    return acc

def test(data,model,test_mask):
    model.eval()
    # Get the output of the model
    out = model(data.x, data.edge_index, data.edge_attr)
    # Compute the accuracy
    pred = out.argmax(dim=1)
    test_correct = pred[test_mask] == data.y[test_mask]
    acc = int(test_correct.sum()) / int(test_mask.sum())
    return acc
