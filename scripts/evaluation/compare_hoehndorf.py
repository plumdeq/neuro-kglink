# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Script to format our results and compare with the SOA (Hoehndorf)

# Plan

* read results csv into a dataframe and compute mean values for all folds per
  parameters
* read mappings for relations
* read soa results
* write comparison (how much we far off, etc)

"""
# Standard-library imports
import os
import logging
from functools import reduce

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import click
import pandas as pd
from sklearn.model_selection import ParameterGrid


class Comparator:
    def __init__(self, results_path, idx2rel_path, rel2name_path, 
                 soa_path, columns=None):
        self.results_path = results_path
        self.idx2rel_path = idx2rel_path
        self.rel2name_path = rel2name_path
        self.soa_path = soa_path
        self.columns = columns or ["dim", "epoch", "relation"]

        self.idx2rel = pd.read_csv(self.idx2rel_path, header=None, index_col=0, sep="\t")
        self.rel2name = pd.read_csv(self.rel2name_path, header=None, index_col=0, sep=" ")
        self.soa_results = pd.read_csv(self.soa_path, index_col=0)

        self.mean_df = self.compute_mean()
        self.hyper_params = self.compute_hyperparams()


    def compute_mean(self):
        """Group by dim, epoch and relation (or other columns) and compute means"""
        df = pd.read_csv(self.results_path)

        # drop "fold" column if any
        if "fold" in df.columns:
            df = df.drop(columns=["fold"])

        mean_df = df.groupby(self.columns).mean().reset_index()

        mean_df["relation"] = mean_df["relation"].apply(self.idx2name)

        return mean_df


    def idx2name(self, idx):
        """
        Gives a map (function) from "id" to relation's name used in the paper

        """
        try:
            return self.rel2name.loc[self.idx2rel.loc[int(idx)].values[0]]
        except Exception as e:
            print("no entry for {}".format(idx))
            raise e
            return None

        return None


    def extract_distinct_values(self, col, df=None):
        """
        Extract a list of distinct values from `df` for column `col`

        """
        # self.mean_df or None does `or` on dataframe and None
        df = self.mean_df if df is None else df

        return df[col].unique().tolist()


    def compute_hyperparams(self, df=None, columns=None):
        """
        Extract all distinct values for the dataframe `df` for the `columns`
        
        """
        # self.mean_df or None does `or` on dataframe and None
        df = self.mean_df if df is None else df
        columns = columns or self.columns

        hyper_params = { col: self.extract_distinct_values(col, df=df) for col in columns }

        return hyper_params


    def extract_subdf(self, params, df=None):
        """
        Extracts a sub dataframe for the given list of parameters (key-value)

        """
        # self.mean_df or None does `or` on dataframe and None
        df = self.mean_df if df is None else df
        selectors = [(df[k] == v) for k, v in params.items()]
        selector = reduce(lambda x, y: x & y, selectors)

        return df[selector]


    def pivot_one_df_with_params(self, params, index_col="relation", df=None):
        """
        Got a given list of key values pivot df around index_col

        """
        # self.mean_df or None does `or` on dataframe and None
        df = self.mean_df if df is None else df

        # create a column name from the parameter list
        col_name = "|".join(["{}-{}".format(k, v) for k, v in sorted(params.items())])

        # columns which we will rename for a given parameter list
        no_cols = [index_col] + list(params.keys())
        cols = [col for col in df.columns if not col in no_cols]

        sub_df = self.extract_subdf(params).set_index(index_col)[cols]
        new_cols = ['|'.join([col_name, c]) for c in sub_df.columns]
        sub_df.columns = new_cols

        return sub_df
        

    def pivot_df(self, index_col="relation", hyper_params=None, df=None):
        """
        Pivot dataframe around index_column, which should also be part of
        `hyper_params`. Returns a new dataframe

        """
        hyper_params = hyper_params or self.hyper_params
        # self.mean_df or None does `or` on dataframe and None
        df = self.mean_df if df is None else df

        if not index_col in hyper_params.keys():
            raise Exception("index_col {} not in hyper params".format(index_col))

        df = pd.DataFrame(index=hyper_params[index_col])
        # get hyper params dict without index_col
        hyper_params = { k:v for k, v in hyper_params.items() if not k == index_col }

        # cols = [col for col in hyper_params.keys() if not col == index_col]

        dfs = [self.pivot_one_df_with_params(params) for params in ParameterGrid(hyper_params)]

        return pd.concat(dfs, axis=1)


    def compute_cols_to_compare(self, cols, col_to_compare):
        """Select columns from `cols` that (substring) match `col_to_compare`"""
        cols_to_keep = []

        return [c for c in cols if not c.find(col_to_compare) == -1]


    def compare_with_soa(self, cols_to_compare=["F-measure", "ROC AUC"]):
        """
        Writes out the summary of the obtained results with that of the SOA

        """
        # for col_to_compare in cols_to_compare:
        #     print("comparison for {}".format(col_to_compare))

        #     idx_max = self.mean_df.groupby("relation")[col_to_compare].transform(max) == self.mean_df[col_to_compare]

        #     max_df = self.mean_df[idx_max].copy()
        #     max_df = max_df.set_index("relation")

        #     max_df[col_to_compare + "_SOA_delta"] = \
        #         max_df[col_to_compare].subtract(self.soa_results[col_to_compare],
        #                                         fill_value=0.0)
        #     
        #     name = col_to_compare.lower().replace(" ", "_")
        #     name = name + "-results.csv"
        #     max_df.to_csv(os.path.join(self.out_dir, name))
        #     

        # return None
        pivoted_df = self.pivot_df()

        # collect dfs with subtracted info
        dfs = []
        for col in self.soa_results.columns:
            cols_to_compare = self.compute_cols_to_compare(
                    pivoted_df.columns, col)

            dfs.append(pivoted_df[cols_to_compare].apply(
                lambda s: s.subtract(self.soa_results[col], fill_value=0.0)))

        return pd.concat(dfs, axis=1)


    def all_better(self):
        """Return configurations where all relations are better than soa
        results"""
        df = self.compare_with_soa()
        idxs = [all(df[col] > 0) for col in df.columns]

        return df.iloc[:, idxs]



@click.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.argument("idx2rel_path", type=click.Path(exists=True))
@click.argument("rel2name_path", type=click.Path(exists=True))
@click.argument("soa_path", type=click.Path(exists=True))
@click.option("--out-dir", type=click.Path(exists=True), default=".",
              help="directory where we store results")
@click.option("--out-path", default="comparison.csv", help="name of the output file")
def main(results_path, idx2rel_path, rel2name_path, soa_path, out_dir, out_path):
    comparator = Comparator(results_path, idx2rel_path, rel2name_path, soa_path)

    click.echo("comparing our results with SOA")
    result_df = comparator.compare_with_soa()

    path = os.path.join(out_dir, out_path)
    result_df.to_csv(path, float_format="%.3g")


if __name__ == "__main__":
    main()
