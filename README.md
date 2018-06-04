# Evaluation of neural embeddings for link prediction tasks

We focus our attention on the link prediction problem for knowledge graphs,
which is treated herein as a binary classification task on neural embeddings of
the entities. By comparing, combining and extending different methodologies for
link prediction on graph-based data coming from different domains, we formalize
a unified methodology for the quality evaluation benchmark of neural embeddings
for knowledge graphs. This benchmark is then used to empirically investigate
the potential of training neural embeddings globally for the entire graph, as
opposed to the usual way of training embeddings locally for a specific
relation. This new way of testing the quality of the embeddings evaluates the
performance of binary classifiers for scalable link prediction with limited
data. Our evaluation pipeline is made open source to the community, and with
this we aim to draw more attention of the community towards an important issue
of transparency and reproducibility of the neural embeddings evaluations.

## Requirements

* non-Python requirements 
    * `starspace`
* Python requirements
    * click==6.7
    * tqdm==4.19.4
    * pandas==0.21.0
    * numpy==1.13.3
    * scipy==1.0.0
    * scikit-learn==0.19.1

## Usage

We prepared a set of scripts for: 

* statistics about the links of the knowledge graph
* data generation (unsupervised corpus for neural embedding training), train
  and test splits for binary classifiers
* neural embedding training with StarSpace (requires `starspace`)
* evaluation of the neural embeddings on the train and test splits

All scripts can be found in `./scripts` folder. Note that all the python
scripts use relative imports, i.e., you should not move them from that folder
(we will make the more path agnostic later). For help use `--help` option when
you call a specific script. Input KG is assumed to be in the `head_entity, relation,
tail_entity` format, one entry per line.

### KG statistics

```
python3 ./scripts/kg_stats.py [OPTIONS] GRAPH_PATH
``` 

Will read in the KG in `GRAPH_PATH` and display statistics about each relation,
the result -- a pandas dataframe -- is written to `--out-file` 

### Data generation

```
python3 ./scripts/data_generation/lp_data.py [OPTIONS] INPUT_PATH
``` 

Will create a file structure as the one shown below inside the `--output-path`,
global graphs will be used for training embeddings globally. Subfolders
correspond each to a specific relation and inside each folder you will have
unsupervized corpus, train and test splits for as many folds as you specify
(`--folds`).

```
.
├── 40943
|   ...
├── 40953
│   ├── relation-40953-test_biased_single_label-fold-1
|   ...
│   ├── relation-40953-test_biased_single_label-fold-10
│   ├── relation-40953-test_graph-fold-1
|   ...
│   ├── relation-40953-test_graph-fold-10
│   ├── relation-40953-test_unbiased_single_label-fold-1
|   ...
│   ├── relation-40953-test_unbiased_single_label-fold-10
│   ├── relation-40953-train_biased_single_label-fold-1
│   ...
│   ├── relation-40953-train_biased_single_label-fold-10
│   ├── relation-40953-train_graph-fold-1
|   ...
│   ├── relation-40953-train_graph-fold-10
│   ├── relation-40953-train_unbiased_single_label-fold-1
|   ...
│   └── relation-40953-train_unbiased_single_label-fold-10
├── ent2idx.map
├── global-test-fold-1
...
├── global-test-fold-9
...
├── global-train-fold-9
├── rel2idx.map
└── stats.csv
```

### Training embeddings

```
python3 ./scripts/train_embeddings/train_mode{1,2}.py [OPTIONS] INPUT_DIR STARSPACE_BIN
```

Mode 1 corresponds to global embeddings training, and mode 2 will train local
to each relation embeddings. Both scripts will train embeddings with
`starspace` (installed on your system, `STARSPACE_BIN` should point to its
binary). `INPUT_DIR` is the folder with the generated data. `--output-dir` will
contain embeddings for the specified hyperparameters in the `--params-file'
(JSON file with parameter values).

```
.
├── 40943
|   ...
├── 40953
├── execution_times.csv
├── external.log
├── global-train-fold-10-dim-10_epoch-10
├── global-train-fold-10-dim-10_epoch-10.tsv
|   ...
├── global-train-fold-10-dim-50_epoch-5
└── global-train-fold-10-dim-50_epoch-5.tsv
```

### Evaluation

```
python3 ./scripts/train_embeddings/eval-mode{1,2}.py [OPTIONS] EMBEDDINGS_DIR TRAIN_DIR
``` 

Same as training embeddings you can either evaluate globally for the whole KG
(mode 1) or locally for a specific relation (mode 2). Pandas dataframes with
the evaluation results will be written to `--output-dir`.
