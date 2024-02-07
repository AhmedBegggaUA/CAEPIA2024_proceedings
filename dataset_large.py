import torch
import gdown
import pandas as pd
from os import path
import os.path as osp
from torch_geometric.data import Dataset, download_url, Data
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
import scipy
import scipy.io
DATAPATH = "./data/"
dataset_drive_url = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',  # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',  # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'  # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}
def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label
def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features
def twitch_gamer():
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
                        output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):                    
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
                        output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task="mature")
    node_feat = torch.tensor(features, dtype=torch.float)
    normalize = False
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    label = torch.tensor(label)
    dataset = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return dataset

def genius():
    fulldata = scipy.io.loadmat(f'{DATAPATH}genius.mat')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = len(node_feat)
    dataset = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return dataset

def pokec():
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'],
                        output=f'{DATAPATH}pokec.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    normalize = False
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    label = fulldata['label'].flatten()
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label)
    return data
def snap_patents():
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
                       output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = node_feat.shape[0]
    years = fulldata['years'].flatten()
    label = even_quantile_labels(vals=years, nclasses=5, verbose=False)
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data
def arxiv_year():
    ogb_dataset = NodePropPredDataset(root='./data', name='ogbn-arxiv')
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])

    nclass = 5
    label = even_quantile_labels(
        ogb_dataset.graph['node_year'].flatten(), nclass, verbose=False)
    label = torch.as_tensor(label).reshape(-1, 1).squeeze()
    data = Data(x=node_feat, edge_index=edge_index, y=label)
    return data
def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx
