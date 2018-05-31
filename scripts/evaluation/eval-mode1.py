# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Evaluate given embeddings with a logistic regression classifier. The evaluation
mode "1" is 'single graph and multiple single label tasks'. That is, we train the
embeddings once on the overall big graph, and then we evaluate for each
specific relation.

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

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, path)

# Cross-library imports
from kglink import kg_evaluator


class MassEvaluator(object):
    def __init__(self, evaluator, embeddings_dir=None, train_dir=None, test_dir=None, 
                 output_dir=None, output_fname=None, regexp_params=None, regexp_fold=None, 
                 regexp_embeddings=None, regexp_train=None, regexp_test=None,
                 binary_op=None, dry_run=None):
        self.evaluator         = evaluator
        self.embeddings_dir    = embeddings_dir
        self.train_dir         = train_dir
        self.test_dir          = test_dir
        self.output_dir        = output_dir
        self.output_fname      = output_fname
        self.regexp_fold       = regexp_fold
        self.regexp_params     = regexp_params
        self.regexp_embeddings = regexp_embeddings
        self.regexp_train      = regexp_train
        self.regexp_test       = regexp_test
        self.binary_op         = binary_op
        self.dry_run           = dry_run


    def collect_subfolders(self, path):
        root, subfolders, _ = next(os.walk(path))
        return { subfolder: os.path.join(root, subfolder) 
                 for subfolder in subfolders }


    def collect_files(self, folder, filename_pattern, fold_pattern=None):
        """
        Will create a map 'fold -> [file]'
        
        Goes through all files and matches with a pattern 'fold-\d', then uses
        the matched number as the value, and the name of the file as the key
        
        """
        fold_pattern = fold_pattern or self.regexp_fold

        file_map = defaultdict(list)
        pattern = os.path.join(folder, filename_pattern)
        files = glob.glob(pattern)
        logger.info("creating embedding files map fold -> [embedding file]")
        logger.info("will scan files with pattern {}".format(filename_pattern))

        for f in files:
            match = re.match(fold_pattern, f)
            try:
                fold = match.group(1)
                file_map[fold].append(f)
            except Exception as e:
                pass

        return file_map


    def extract_params(self, fname):
        """Extract param names and values from the filename. Assume that the
        filename follows 'dim-10-epoch-10'"""
        return re.findall(self.regexp_params, os.path.basename(fname))


    def bulk_evaluate(self):
        """Collect subfolders corresponding to relations, and evaluate per fold and
        per relation"""
        emb_files = self.collect_files(
                self.embeddings_dir, self.regexp_embeddings, self.regexp_fold)
        logger.info("Collecting embeddings files")
        logger.info(emb_files)

        logger.info("collecting train and test files")
        relations = self.collect_subfolders(self.train_dir)
        logger.info(relations)

        results = []

        try:
            for relation in tqdm(relations, desc="relations"):
                logger.info("relation is {}".format(relation))
                path = relations[relation]
                tr_files = self.collect_files(path, self.regexp_train, self.regexp_fold)
                logger.info(tr_files)
                te_files = self.collect_files(path, self.regexp_test, self.regexp_fold)
                logger.info(te_files)
                # tr_files and te_files is { '1': ['/path/xpath/] }
                for fold in tqdm(tr_files, desc="folds"):
                    tr_file = tr_files[fold][0]
                    te_file = te_files[fold][0]
                    logger.info("tr file is {}, te file is {}".format(tr_file, te_file))

                    for emb_file in tqdm(emb_files[fold], desc="embeddings"):
                        params = dict(self.extract_params(emb_file), relation=relation)
                        logger.info("params are {}".format(params))
                        logger.info("binary op {}".format(self.binary_op))
                        result = self.evaluator.evaluate_params(
                                emb_file, tr_file, te_file, params, binary_op=self.binary_op)
                        results.append(result)
        except KeyboardInterrupt:
            logger.info("Returning partial results")
            return results

        return results


    def write_results(self, results):
        """Write results to a file"""
        if not os.path.exists(self.output_dir):
            logger.info("creating {} (did not exist)".format(self.output_dir))
            os.makedirs(self.output_dir)

        out_file = os.path.join(self.output_dir, self.output_fname)
        logger.info("writing to {}".format(out_file))
        df = pd.DataFrame(results)
        df.to_csv(out_file, index=False)
            
        return None


@click.command()
@click.argument("embeddings_dir", type=click.Path(exists=True))
@click.argument("train_dir", type=click.Path(exists=True))
@click.option("--test_dir", type=click.Path(exists=True))
@click.option("--output-dir", default="./evaluations", help="directory where output is written to")
@click.option("--output-fname", default="results-global.csv", help="filename of the csv file with results")
@click.option("--regexp-fold", default=r'.*fold-(\d+).*')
@click.option("--regexp-params", default=r'([a-zA-Z]+)-(\d+)')
@click.option("--regexp-embeddings", default=r'*.tsv', help="glob pattern for finding embedding files")
@click.option("--regexp-train", default=r'*train*', help="glob pattern for finding train files")
@click.option("--regexp-test", default=r'*test*', help="glob pattern for finding test files")
@click.option("--binary-op", default='concat', type=click.Choice(["concat", "mean", "sum", "mult"]))
@click.option("--dry-run", default=False, is_flag=True)
def main(embeddings_dir, train_dir, test_dir, output_dir, output_fname,
        regexp_fold, regexp_params, regexp_embeddings, regexp_train,
        regexp_test, binary_op, dry_run):

    test_dir = test_dir or train_dir

    my_evaluator = kg_evaluator.KGEvaluator()

    mass_evaluator = MassEvaluator(
            evaluator=my_evaluator,
            embeddings_dir=embeddings_dir, train_dir=train_dir,
            test_dir=test_dir, output_dir=output_dir, 
            output_fname=output_fname, regexp_fold=regexp_fold, 
            regexp_params=regexp_params,
            regexp_embeddings=regexp_embeddings, 
            regexp_train=regexp_train, 
            regexp_test=regexp_test,
            binary_op=binary_op,
            dry_run=dry_run)

    results = mass_evaluator.bulk_evaluate()
    mass_evaluator.write_results(results)


if __name__ == '__main__':
    main()
