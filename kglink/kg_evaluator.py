# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Evaluate given embeddings with a logistic regression classifier

"""
# Standard-library imports
import re
import os
import sys
import logging
import glob
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
from tqdm import tqdm
import numpy as np
from sklearn import linear_model, neural_network
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
import click
import pandas as pd

# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.insert(0, path)

# Cross-library imports
from . import utils


class KGEvaluator(object):
    """Collects embedding files, associates them with the matching pairs of
    train/test files for single-label classification task, and evaluates these
    embeddings with a logistic regression"""
    def __init__(self):
        # representation functions for one arc from two embeddings of a node
        self.repr_fns = {
            "sum": lambda x, y: x + y,
            "mean": lambda x, y: (x + y) / 2,
            "mult": lambda x, y: x * y,
            "concat": lambda x, y: np.concatenate([x, y]),
        }


    def get_embeddings(self, emb_file):
        # build dictionary "idx: x1 x2"
        logger.info("reading embeddings from {}".format(emb_file))
        E = {}

        with open(emb_file, "r") as f:
            for l in f.readlines():
                line = l.strip().split() 
                E[line[0]] = np.array(list(map(float, line[1:]))).astype(np.float32)

        return E


    def convert_data(self, f, E, binary_op="concat"):
        """read in `v1 v2 label` and train on <E(v1), E(v2)> = 1|0"""

        if binary_op not in self.repr_fns:
            logger.info("{} - unknown representation, falling back to concatenation".format(binary_op))

        repr_fn = self.repr_fns.get(binary_op, self.repr_fns["concat"])
        size = len(E[list(E.keys())[0]])
        
        with open(f, "r") as f:
            lines = f.readlines()

            N = len(lines)

            # take a sample to determine the length of the embeddings
            v1, v2, label = lines[0].strip().split()

            # adjust array size to the representation type
            d = size if binary_op != "concat" else size*2
            X = np.zeros((N, d)) 
            y = np.ones(N).astype(np.uint16)

            # collect how many missing embeddings we get
            missing = []
            # read in v1, v2, labels and compute X, y
            for i, l in enumerate(lines):
                v1, v2, label = l.strip().split()

                # if no embedding found that delete this example
                try:
                    emb = repr_fn(E[v1], E[v2])

                    X[i] = emb
                    y[i] = int(label)

                except Exception as e:
                    missing.append(i)

        # delete examples with missing embeddings
        if len(missing) > 0:
            X = np.delete(X, missing, 0)
            y = np.delete(y, missing, 0)

        assert i == N-1, "i={}, N={}".format(i, N)

        missing_ratio = len(missing)/N*100
        logger.info("missing embeddings {0}/{1} {2:0.3f}%".format(len(missing), N, missing_ratio))
        logger.info("X shape {}, y shape {}".format(X.shape, y.shape))

        return X, y, missing_ratio


    def evaluate_log_regression(self, data):
        """
        Log regression and MLP should be refactored, as the interface to *train,
        fit and predict* is the same for both. What changes are different
        parameters

        """
        logger.info("applying logistic regression")
        X_tr, X_te, y_tr, y_te = data

        logistic = linear_model.LogisticRegression()
        logistic.fit(X_tr, y_tr)
        mean_acc = logistic.score(X_te, y_te)

        predicted = logistic.predict(X_te)

        f_measure = f1_score(y_te, predicted)
        roc_auc = roc_auc_score(y_te, predicted)
        
        return {
            "Mean acc": mean_acc,
            "F-measure": f_measure,
            "ROC AUC": roc_auc,
            }


    def evaluate_mlp(self, data, hidden_layer_sizes=(200,)):
        """
        TODO: REFACTOR

        """
        logger.info("applying MLP")
        X_tr, X_te, y_tr, y_te = data

        mlp = neural_network.MLPClassifier(solver="lbfgs", alpha=1e-5, 
                                           hidden_layer_sizes=hidden_layer_sizes, 
                                           random_state=1)
        mlp.fit(X_tr, y_tr)
        mean_acc = mlp.score(X_te, y_te)

        predicted = mlp.predict(X_te)

        f_measure = f1_score(y_te, predicted)
        roc_auc = roc_auc_score(y_te, predicted)
        
        return {
            "Mean acc": mean_acc,
            "F-measure": f_measure,
            "ROC AUC": roc_auc,
            }


    def log_data(self, data, n=10):
        X_tr, X_te, y_tr, y_te = data

        logger.info("How balanced are classes")
        logger.info("*"*20)
        logger.info("train data: #0 {}, #1 {}".format(
            sum(y_tr == 0), sum(y_tr == 1)))
        logger.info("test data: #0 {}, #1 {}".format(
            sum(y_te == 0), sum(y_te == 1)))

        # randomize indices for train and test data for logging purposes only
        random_idx_tr = list(range(len(X_tr)))
        random_idx_te = list(range(len(X_te)))
        random.shuffle(random_idx_tr)
        random.shuffle(random_idx_te)

        logger.info(X_tr[random_idx_tr[:n]])
        logger.info(y_tr[random_idx_tr[:n]])
        logger.info("-"*20)
        logger.info(X_te[random_idx_te[:n]])
        logger.info(y_te[random_idx_te[:n]])


    def evaluate_params(self, emb_file, train_file, test_file, params,
                        binary_op='concat', classifier_type='log_reg'):
        """
        Evaluate takes a callable for file names, hyperparameter dictionary, the
        representation type of the embedding, and the classifier type. It will then
        decide which hyperparameters to pass to which classifier etc.

        """
        E = self.get_embeddings(emb_file)

        X_tr, y_tr, tr_miss_ratio = self.convert_data(train_file, E, binary_op=binary_op) 
        X_te, y_te, te_miss_ratio = self.convert_data(test_file, E, binary_op=binary_op) 

        scores = None
        data = (X_tr, X_te, y_tr, y_te)

        try:
            if classifier_type == "log_reg":
                scores = self.evaluate_log_regression(data)
            elif classifier_type == "mlp": 
                if "hidden_layer_sizes" in params:
                    scores = self.evaluate_mlp(data, params["hidden_layer_sizes"])
                else:
                    scores = self.evaluate_mlp(data)
            else:
                raise Exception("Unknown classifier type {}".format(classifier_type))
        except Exception as e:
            logger.info("Could not evaluate")
            logger.info(e)
            scores = {
                "Mean acc": np.nan,
                "F-measure": np.nan,
                "ROC AUC": np.nan,
            }

        logger.info(",".join(["{}: {}".format(k,v) for k, v in params.items()]))

        for metric, score in scores.items():
            logger.info("{0} on test data {1:0.3f}".format(metric, score))

        return {
            **scores,
            **params,
            "tr_miss_ratio": tr_miss_ratio,
            "te_miss_ratio": te_miss_ratio,
        }
