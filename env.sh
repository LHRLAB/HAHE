#!/bin/bash

conda create -n hahe -y python=3.8
conda activate hahe

conda install scikit-learn -y
conda install numpy -y
conda install pandas -y
conda install tqdm -y

pip install torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl
pip install -U torch_geometric
