# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Makes subprocess calls to train embeddings with StarSpace. This script takes as
input a configuration file, which consists of parameter grids - a dictionary
where values are lists of possible parameter values for hyperparameter
optimization.

Note that this script only deals with SINGLE RETAINED MULTIPLE SINGLE LABEL
TASKS problem.

"""
# Standard-library imports
import shlex
import os
import subprocess
import glob
import logging
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Third-party imports
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import click
import pandas as pd


path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, path)

# Cross-library imports
from kglink import utils


STARSPACE_OPTS = 'train -trainMode 1 -similarity dot -label ""'
HYPER_PARAMS = {
    "epoch": [10],
    "dim": [5, 10, 20, 50],
}
EXTERNAL_LOG = 'external_global.log'
EXECUTION_TIMES_LOG = 'execution_times_global.csv'


class Trainer(object):
    """Collects train and test files for different folds and trains embeddings
    with StarSpace for all those folds"""
    def __init__(self, input_dir, output_dir, regexp_train, starspace_bin,
                 param_file=None, starspace_opts=None,
                 external_log=EXTERNAL_LOG,
                 execution_times_log=EXECUTION_TIMES_LOG,
                 dry_run=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.regexp_train = regexp_train
        self.param_file = param_file
        self.starspace_opts = starspace_opts or STARSPACE_OPTS
        self.starspace_bin = starspace_bin
        self.external_log = external_log
        self.external_log = os.path.join(output_dir, self.external_log)
        self.execution_times_log = execution_times_log
        self.execution_times_log = os.path.join(output_dir, self.execution_times_log)
        self.dry_run = dry_run

        self.train_files = self.collect_files()
        self.hyper_params = self.get_hyperparams()

        # will contain a DataFrame of execution times
        self.execution_times = []


    def collect_files(self):
        train_files = glob.glob(self.input_dir + self.regexp_train + '*')
        logger.info("train files are {}".format(train_files))

        return train_files


    def get_hyperparams(self):
        """Collect hyperparameters"""
        hyper_params = {}
        if not self.param_file:
            logger.info("using default hyper params:")
            logger.info(HYPER_PARAMS)

            hyper_params = dict(**HYPER_PARAMS)
            hyper_params['graph'] = self.train_files

            return hyper_params

        logger.info("reading hyperparams from {}".format(self.param_file))

        try:
            with open(self.param_file, 'r')  as f:
                hyper_params = dict(**json.load(f))

            logger.info("hyperparams are:")
            logger.info(hyper_params)

            hyper_params = dict(**hyper_params)
            hyper_params['graph'] = self.train_files

            return hyper_params

        except Exception as e:
            sys.exit(e)


    def compute_model_name(self, params):
        """Composes output model name from parameters (e.g., epoch, dim)"""
        prefix = os.path.basename(params['graph'])
        name = "_".join(["{}-{}".format(k, v) 
                        for k, v in sorted(params.items())
                        if k not in ["graph"]])

        model_name = '-'.join([prefix, name])
        full_name = os.path.join(self.output_dir, model_name)

        return full_name


    @utils.timeit
    def call_starspace(self, params, train_file, model_name):
        """Launch external subprocess which will call StarSpace"""
        logger.info("calling '{}' with options: {}".format(self.starspace_bin, self.starspace_opts))
        cmd_string = ["-{0} {1}".format(k, v) for k, v in params.items() if k not in ['graph']]
        cmd_string.append("-trainFile {}".format(train_file))
        cmd_string.append("-model {}".format(model_name))

        cmd_arg_list = " ".join([self.starspace_bin, self.starspace_opts]) + " " + " ".join(cmd_string)

        logger.info("cmd arg list is {}".format(cmd_arg_list))

        if not self.dry_run:
            cmd_output = subprocess.run(shlex.split(cmd_arg_list), stdout=subprocess.PIPE)
            return cmd_output.stdout.decode("utf-8")
        else:
            logger.info("dry run")
            return "dry run"


    def train_embeddings(self):
        """Loop through all hyperparams and train StarSpace embeddings"""
        logger.info("training embeddings for files {}".format(self.train_files))
        logger.info("all output will be written to {}".format(self.output_dir))

        if not os.path.exists(self.output_dir):
            logger.info("creating {} (did not exist)".format(self.output_dir))
            os.makedirs(self.output_dir)

        for params in tqdm(ParameterGrid(self.hyper_params), desc="training embedding"):
            logger.info("hyperparams: {}".format(params))
            train_file = params['graph']
            model_name = self.compute_model_name(params)
            logger.info('training starspace model "{}" from file "{}"'.format(
                model_name, train_file))
            external_output, delta = self.call_starspace(params, train_file, model_name)
            logger.info("executed in {:0.2f}s".format(delta))

            logger.info("external command output logged in {}".format(self.external_log))
            with open(self.external_log, 'a') as f:
                f.write(external_output)

            self.execution_times.append(dict({ 'time': delta }, **params))

        logger.info("writing execution times log into {}".format(
            self.execution_times_log)) 
        df = pd.DataFrame(self.execution_times)
        df.to_csv(self.execution_times_log, index=False)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("starspace-bin", type=click.Path(exists=True))
@click.option("--output-dir", default="./embeddings")
@click.option("--regexp-train", default="*train*")
@click.option("--param-file", default=None)
@click.option("--starspace-opts", default=None)
@click.option("--external-log", default=EXTERNAL_LOG)
@click.option("--execution-times-log", default=EXECUTION_TIMES_LOG)
@click.option("--dry-run", default=False, is_flag=True)
def dev_test(input_dir, starspace_bin, output_dir, regexp_train, param_file,
             starspace_opts, external_log, execution_times_log, dry_run):
    trainer = Trainer(input_dir, output_dir, regexp_train,
                        param_file=param_file, starspace_bin=starspace_bin,
                        starspace_opts=starspace_opts,
                        external_log=external_log,
                        execution_times_log=execution_times_log,
                        dry_run=dry_run)
    trainer.train_embeddings()

if __name__ == "__main__":
    # train()
    dev_test()
