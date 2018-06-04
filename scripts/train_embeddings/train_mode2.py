# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Makes subprocess calls to train embeddings with StarSpace. This script takes as
input a configuration file, which consists of parameter grids - a dictionary
where values are lists of possible parameter values for hyperparameter
optimization.

Note that this script only deals with MULTIPLE RETAINED MULTIPLE SINGLE LABEL
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


STARSPACE_OPTS = 'train -trainMode 1 -similarity dot -label "" -maxNegSamples 50'
HYPER_PARAMS = {
    "epoch": [10],
    "dim": [5, 10, 20, 50],
}
EXTERNAL_LOG = 'external_local.log'
EXECUTION_TIMES_LOG = 'execution_times_local.csv'


class Trainer(object):
    """Collects train and test files for different folds and trains embeddings
    with StarSpace for all those folds"""
    def __init__(self, input_dir, output_dir, regexp_train, starspace_bin=None,
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


    def collect_subfolders(self, path):
        root, subfolders, _ = next(os.walk(path))
        return { subfolder: os.path.join(root, subfolder) 
                 for subfolder in subfolders }


    def collect_files(self, folder=None, fname_pattern=None):
        folder = folder or self.input_dir
        fname_pattern = fname_pattern or self.regexp_train

        files = glob.glob(folder + '/' + fname_pattern)
        logger.info("collected files {}".format(files))

        return files


    def extract_params(self, fname):
        """Extract param names and values from the filename. Assume that the
        filename follows 'dim-10-epoch-10'"""
        return re.findall(self.regexp_params, os.path.basename(fname))


    def get_hyperparams(self):
        """Collect hyperparameters"""
        hyper_params = {}
        if not self.param_file:
            logger.info("using default hyper params:")
            logger.info(HYPER_PARAMS)

            hyper_params = dict(**HYPER_PARAMS)
            # hyper_params['graph'] = self.train_files

            return hyper_params

        logger.info("reading hyperparams from {}".format(self.param_file))

        try:
            with open(self.param_file, 'r')  as f:
                hyper_params = dict(**json.load(f))

            logger.info("hyperparams are:")
            logger.info(hyper_params)

            hyper_params = dict(**hyper_params)
            # hyper_params['graph'] = self.train_files

            return hyper_params

        except Exception as e:
            sys.exit(e)


    def compute_model_name(self, params, folder=None):
        """Composes output model name from parameters (e.g., epoch, dim)"""
        folder = folder or self.output_dir

        prefix = os.path.basename(params['graph'])
        name = "_".join(["{}-{}".format(k, v) 
                        for k, v in sorted(params.items())
                        if k not in ["graph"]])

        model_name = '-'.join([prefix, name])
        full_name = os.path.join(folder, model_name)

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


    def bulk_train(self):
        """Bulk train embeddings for all relations in the given input folder"""
        logger.info("collecting subfolders - relations")
        relations = self.collect_subfolders(self.input_dir)
        logger.info("relations - {}".format(relations))

        execution_times = []

        for rel, rel_path in tqdm(relations.items(), desc="relations"):
            logger.info("collecting training files from {}".format(rel_path))
            tr_files = self.collect_files(rel_path, self.regexp_train)
            hyper_params = self.get_hyperparams()
            hyper_params['graph'] = tr_files

            output_folder = os.path.join(self.output_dir, rel)
            if not os.path.exists(output_folder):
                logger.info("creating {} (did not exist)".format(output_folder))
                os.makedirs(output_folder)

            for params in tqdm(ParameterGrid(hyper_params), desc="training embedding"):
                logger.info("hyperparams: {}".format(params))
                train_file = params['graph']
                model_name = self.compute_model_name(params, output_folder)
                logger.info('training starspace model "{}" from file "{}"'.format(
                    model_name, train_file))
                external_output, delta = self.call_starspace(params, train_file, model_name)
                logger.info("executed in {:0.2f}s".format(delta))

                logger.info("external command output logged in {}".format(self.external_log))
                if not os.path.exists(self.output_dir):
                    logger.info("creating {} (did not exist)".format(self.output_dir))
                    os.makedirs(self.output_dir)

                with open(self.external_log, 'a') as f:
                    f.write(external_output)

                execution_times.append(dict({ 'time': delta }, **params))
                                
        return execution_times


    def write_results(self, results, fname, folder=None):
        """Write results to a file"""
        folder = folder or self.output_dir

        if not os.path.exists(folder):
            logger.info("creating {} (did not exist)".format(folder))
            os.makedirs(folder)

        out_file = os.path.join(folder, fname)
        logger.info("writing to {}".format(out_file))
        df = pd.DataFrame(results)
        df.to_csv(out_file, index=False)
            
        return None


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("starspace-bin", type=click.Path(exists=True))
@click.option("--output-dir", default="./embeddings")
@click.option("--regexp-train", default="*train*graph*")
@click.option("--param-file", default=None)
@click.option("--starspace-opts", default=None)
@click.option("--external-log", default=EXTERNAL_LOG)
@click.option("--execution-times-log", default=EXECUTION_TIMES_LOG)
@click.option("--dry-run", default=False, is_flag=True)
def train(input_dir, starspace_bin, output_dir, regexp_train, param_file, 
             starspace_opts, external_log, execution_times_log, dry_run):
    trainer = Trainer(input_dir, output_dir, regexp_train,
                        param_file=param_file, starspace_bin=starspace_bin,
                        starspace_opts=starspace_opts,
                        external_log=external_log,
                        execution_times_log=execution_times_log,
                        dry_run=dry_run)
    execution_times = trainer.bulk_train()
    trainer.write_results(execution_times, execution_times_log)


if __name__ == "__main__":
    train()
