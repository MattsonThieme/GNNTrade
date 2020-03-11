#!/bin/bash
yes 'y' | conda create -n gnntrade -c intel python=3
source activate gnntrade
yes 'y' | conda install numpy
yes 'y' | conda install pytorch torchvision -c pytorch
pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster
pip install torch-geometric
pip install ccxt
pip install tqdm
source deactivate
