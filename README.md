# HEX-GNNs
HEX-GNN: Hierarchical EXpanders for Node
Classification
## Dependencies

Conda environment
```
conda create --name <env> --file requirements.txt
```

or

```
conda env create -f conda_graphy_environment.yml
conda activate graphy
```
## Code organization
* `data/`: folder with the datasets.
* `logs/`: folder with the logs of the experiments.
* `runners/`: folder with the scripts for running the experiments.
* `splits/`: splits that we used, taking from GEO-GCN repository.
* `main.py`: script with inline arguments for running the experiments.
* `main_large.py`: script with inline arguments for running large experiments.
* `models.py`: script with our proposed architecture.
* `utils.py`: extra functions used for the experiments.
* `dataset_large.py`: script with the dataset class for large datasets.
## Run experiments
```python
python main.py --dataset wisconsin --hidden_channels 16 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.4 --wd 0.001 
python main.py --dataset texas --hidden_channels 16 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.4 --wd 0.001  
python main.py --dataset cornell  --hidden_channels 512 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.35 --wd 0.0005 
python main.py --dataset citeseer --hidden_channels 16 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.5 --wd 0.0008 
python main.py --dataset cora --hidden_channels 32 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.5 --wd 0.001 
python main.py --dataset pubmed --hidden_channels 16 --hops 3 --lr 0.003 --epochs 10000 --dropout 0.5 --wd 0.0005 
python main.py --dataset chamaleon --hidden_channels 16 --hops 3 --lr 0.001 --epochs 10000 --dropout 0.4 --wd 0.0005 
python main.py --dataset squirrel --hidden_channels 256 --hops 3 --lr 0.001 --epochs 10000 --dropout 0.4 --wd 0.0005 
```
