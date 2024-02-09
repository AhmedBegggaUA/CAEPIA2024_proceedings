python main.py --dataset wisconsin --hidden_channels 16 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.4 --wd 0.001 # 89.02,2.8
python main.py --dataset texas --hidden_channels 16 --hops 6 --lr 0.03 --epochs 10000 --dropout 0.4 --wd 0.001  # Report:  89.1891891891892 +- 4.1870090229269366
python main.py --dataset cornell  --hidden_channels 512 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.35 --wd 0.0005 # Report:  85.94594594594595 +- 2.9108998957483796
python main.py --dataset citeseer --hidden_channels 16 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.5 --wd 0.0008 #Report:  77.87176327270667 +- 1.5163752309719407
python main.py --dataset cora --hidden_channels 32 --hops 3 --lr 0.03 --epochs 10000 --dropout 0.5 --wd 0.001 #Report:  88.61167002012073 +- 0.9573744268306076 
python main.py --dataset pubmed --hidden_channels 16 --hops 3 --lr 0.003 --epochs 10000 --dropout 0.5 --wd 0.0005 # Report:  89.78194726166329 +- 0.38502838281395774
python main.py --dataset chamaleon --hidden_channels 256 --hops 4 --lr 0.001 --epochs 10000 --dropout 0.4 --wd 0.0005 # Report:  51.42543859649123 +- 1.513793408370378
python main.py --dataset squirrel --hidden_channels 256 --hops 4 --lr 0.001 --epochs 10000 --dropout 0.4 --wd 0.0005 # Report:  38.059558117195 +- 1.2706286850172115
