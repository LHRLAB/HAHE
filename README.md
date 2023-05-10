# HAHE: Hierarchical Attention for Hyper-Relational Knowledge Graphs
## Introduction

This is the [Pytorch](https://pytorch.org/) implementation of HAHE, a novel Hierarchical Attention model for
HKG Embedding, including global-level and local-level attentions. 

This repository contains the code and data, as well as the optimal configurations to reproduce the reported results.

## Requirements and Installation
This project should work fine with the following environments:
- Python 3.8.13 for data preprocessing, training and evaluation with:
    -  torch 1.10.0+cu113
    -  toch-geometric 2.1.0.post1
    -  torch-scatter 2.0.9
    -  torch-sparse 0.6.13
    -  numpy 1.23.3
- GPU with CUDA 11.3

All the experiments are conducted on a single 11G NVIDIA GeForce 1080Ti.

#### Setup with Conda

```
bash env.sh
```

## How to Run

### Step 1. Download raw data
We consider three representative n-ary relational datasets, and the datasets can be downloaded from:
- [JF17K](https://www.dropbox.com/sh/ryxohj363ujqhvq/AAAoGzAElmNnhXrWEj16UiUga?dl=0)
- [WikiPeople](https://github.com/gsp2014/WikiPeople)
- [WD50K](https://zenodo.org/record/4036498#.Yx06qoi-uNz)

### Step 2. Preprocess data
Then we convert the raw data into the required format for training and evaluation. The new data is organized into a directory named `data`, with a sub-directory for each dataset. In general, a sub-directory contains:
- `train.json`: train set
- `valid.json`: dev set
- `test.json`: test set
- `all.json`: combination of train/dev/test sets, used only for *filtered* evaluation
- `vocab.txt`: vocabulary consisting of entities, relations, and special tokens like [MASK] and [PAD]

> Note: JF17K is the only one that provides no dev set.

### Step 3. Training & Evaluation

To train and evaluate the HAHE model, please run:

```
mkdir ckpts
mkdir results
python -u ./src/run.py --name [TEST_NAME] --device [GPU_ID] -vocab_size [VOCAB_SIZE] --vocab_file [VOCAB_FILE] \
                       --train_file [TRAIN_FILE] --test_file [TEST_FILE] --ground_truth_file [GROUND_TRUTH_FILE] \
                       --num_workers [NUM_WORKERS] --num_relations [NUM_RELATIONS] \
                       --max_seq_len [MAX_SEQ_LEN] --max_arity [MAX_ARITY]
```

Here you should first create two directories to store the parameters and results of HAHE respectively, then you can set parameters of one dataset according to its statisitcs.
`[TEST_NAME]` is the unique name identifying one Training & Evaluation,  `[GPU_ID]` is the GPU ID you want to use.
`[VOCAB_SIZE]` is the number of vocab of the dataset.
`[VOCAB_FILE]` & `[TRAIN_FILE]` & `[TEST_FILE]` & `[GROUND_TRUTH_FILE]` are the paths storing the vocab file("vocab.txt"), train file("train.json"), test file("test.json") and ground truth file("all.json").
`[NUM_WORKERS]` is the number of workers when reading the data.
`[NUM_RELATIONS]` is the number of relations of the dataset.
`[MAX_ARITY]` is the maximum arity of N-arys in the datast, `[MAX_SEQ_LEN]` is the maximum length of N-ary sequences, which is equal to (2 * [MAX_ARITY] - 1).

Please modify those hyperparametes according to your needs and characteristics of different datasets.

Take WD50K as an example. To train and evalute on this dataset using default hyperparametes, please run:

```
python -u ./src/run.py --name "TEST-wd50k" --device "0" --vocab_size 47688 --vocab_file "./data/wd50k/vocab.txt" \
                       --train_file "./data/wd50k/train.json" --test_file "./data/wd50k/test.json" \
                       --ground_truth_file "./data/wd50k/all.json" --num_workers 1 --num_relations 531 \
                       --max_seq_len 63 --max_arity 32
```
If run.py is not modified, you can simply run:

```
python -u ./src/run.py --name "TEST-wd50k"
```
