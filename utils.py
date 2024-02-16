import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np
import psutil
import os   
def khops_graphs_sampler(x, edge_index, k,device,features=True):
    similarity = torch.cdist(x, x, p=2)
    # Normalize between 0 and 1
    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
    hops = list()
    attributes = list()
    N = edge_index.max().item() + 1
    # Create the adjacency matrix
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (N, N)).to(device)
    # Add self loops
    I = torch.sparse_coo_tensor(torch.arange(N).unsqueeze(0).repeat(2, 1), torch.ones(N), (N, N)).to(device)
#    A = A + I
    # Degree matrix
    degrees = torch.sparse.sum(A, dim=1)
    degrees = torch.pow(degrees, -0.5)
    # Get the indices of the diagonal elements
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1).to(device)
    values = degrees.coalesce().values().to(device)
    # Create the sparse diagonal matrix
    D_tilde = torch.sparse_coo_tensor(indices, values, (N, N)).to(device)
    A_tilde = torch.sparse.mm(torch.sparse.mm(D_tilde, A), D_tilde)
    del A
    del I
    del degrees
    del indices
    del values
    del D_tilde
    # Now we sparcify A_tilde using a random coin
    A_tilde = A_tilde.to_dense()
    A_tilde = A_tilde * (torch.rand(A_tilde.size()) > 0.5).float()
    A_tilde = A_tilde.to_sparse()
    return 0

def khop_graphs_sparse(x, edge_index, k,name,device,features=True):
    similarity = torch.cdist(x, x, p=2)
    # Normalize between 0 and 1
    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
    hops = list()
    attributes = list()
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
    hops.append(A_tilde_k.clone().coalesce().indices().to(device))
    # Ahora ponemos los pesos de cada una de las aristas
    #attributes.append(A_tilde_k.clone().coalesce().values().to(device))
    for i in range(k - 1):
        print("Computing k: ", i+1, " of ", k-1)
        if device == 'cpu':
            # Mostramos cuanta memoria ram del sistema estamos usando
            print("Ram memory: ", psutil.virtual_memory().percent, "%")
        else:
            # Mostramos cuanta memoria estamos usando
            print(torch.cuda.memory_allocated(device=device), "out of ", torch.cuda.max_memory_allocated(device=device))
        A_tilde_k = torch.sparse.mm(A_tilde_k, A_tilde)
        # We store those indices that in similarity has a value greater than 0.5
        print("Before pruning: ", A_tilde_k.coalesce().indices().size(1))
        if features:
            indices = A_tilde_k.coalesce().indices().to(device)
            indices = indices[:, similarity[indices[0], indices[1]] >= similarity.mean()]
            # Select only the initial number of edges
            if indices.size(1) > edge_index.size(1):
                indices = indices[:, :edge_index.size(1)]
            print("After pruning: ", indices.size(1))
            hops.append(indices.clone())
            #A_tilde_k = torch.sparse_coo_tensor(indices, torch.ones(indices.size(1)), (N, N)).to(device)
    #    attributes.append(A_tilde_k.clone().coalesce().values().to(device))
    return hops#, attributes        
        
    


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
    optimizer.zero_grad()
    return loss, acc
@torch.no_grad()
def val(data,model,val_mask):
    model.eval()
    # Get the output of the model
    out = model(data.x, data.edge_index, data.edge_attr)
    # Compute the accuracy
    pred = out.argmax(dim=1)
    val_correct = pred[val_mask] == data.y[val_mask]
    acc = int(val_correct.sum()) / int(val_mask.sum())
    return acc
@torch.no_grad()
def test(data,model,test_mask):
    model.eval()
    # Get the output of the model
    out = model(data.x, data.edge_index, data.edge_attr)
    # Compute the accuracy
    pred = out.argmax(dim=1)
    test_correct = pred[test_mask] == data.y[test_mask]
    acc = int(test_correct.sum()) / int(test_mask.sum())
    return acc
