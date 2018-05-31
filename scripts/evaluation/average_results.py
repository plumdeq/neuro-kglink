# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Average results in the dataframe and write out in the output file

"""
# Standard-library imports
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Third-party imports
import click
import pandas as pd


def get_idx2rel(idx2rel_path):
    """Read a map file (2 cols per line) with 'relation -> idx'"""
    lines = None
    idx2rel = {}, {}
    logger.info("reading rel2idx from {}".format(idx2rel_path))
    with open(idx2rel_path, 'r') as f:
        lines = [line.split() for line in f.readlines()]

    idx2rel = { i:r for i, r in lines }

    return idx2rel


def group_results(df, idx2rel_path=None):
    idx2rel = None
    if idx2rel_path:
        try:
            idx2rel = get_idx2rel(idx2rel_path)
        except Exception as e:
            logger.info(e)
    else:
        logger.info("No idx2rel map, relations will have numeric indices")

    columns = ["dim", "epoch", "relation"]
    # we will aggregate mean and std for all columns except 'dim', 'epoch',
    # 'relation' and 'fold'
    agg_columns = [c for c in df.columns if c not in columns + ["fold"]]

    # drop "fold" column if any
    if "fold" in df.columns:
        df = df.drop(columns=["fold"])

    # mean_df = df.groupby(columns).mean().reset_index()

    agg_object = { c: ['mean', 'std'] for c in agg_columns }
    grouped_df = df.groupby(columns).agg(agg_object).reset_index()

    # flatten columns

    cols = [' '.join(c).strip() for c in grouped_df.columns]
    grouped_df.columns = cols

    if idx2rel:
        logger.info("changing relation indices to names")
        grouped_df["relation"] = grouped_df["relation"].apply(lambda x: idx2rel[str(x)])

    return grouped_df


@click.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--out-path", default="./results/averaged.csv", help="name of the output file")
@click.option("--idx2rel", default=None, help="relation 2 idx map")
@click.option("--dry-run", default=False, is_flag=True)
def main(results_path, out_path, idx2rel, dry_run):
    df = pd.read_csv(results_path)
    grouped_df = group_results(df, idx2rel_path=idx2rel)

    logger.info(grouped_df.describe())

    if not dry_run:
        folder = os.path.dirname(out_path)
        if not os.path.exists(folder):
            logger.info("creating {} (did not exist)".format(folder))
            os.makedirs(folder)
        logger.info("writing to {}".format(out_path))
        grouped_df.to_csv(out_path, float_format='%.3g', index=None)


if __name__ == "__main__":
    main()
