# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Statistics on the Knowledge graph.   

* #nodes per relation (source and target)
* #arcs per relation 
* max number of edges 
* ratio of positive to negative 

"""
# Standard-library imports
import os
import sys
import logging
import math
from collections import Counter
import operator as op

# Third-party imports
import click
import numpy as np
import pandas as pd

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, path)


# Cross-library imports
from kglink import kg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_graph(path):
    """Read lines from the graph and initialize the KG object"""
    lines = None
    with open(path, "r") as f:
        lines = [l.strip().split()[:3] for l in f.readlines()]

    KG = kg.KG(lines)

    return KG


def stats_graph(KG, n_most=10):
    """
    Global stats for the whole graph

    TODO: for each pair, check if the different relations connecting them are
    distinct relations, i.e., not a simple duplication of a relation, (s1, p1,
    o1), (s1, p1, o1), ...
    
    """
    arcs_per_relation = KG.get_arcs_per_relation()
    most_common_relations = sorted(arcs_per_relation.items(), key=op.itemgetter(1), reverse=True)
    arcs_counts = list(arcs_per_relation.values())
    logger.info("max arcs {}, min arcs {}, median arcs {}".format(
        np.max(arcs_counts), np.min(arcs_counts), np.median(arcs_counts)))
    logger.info("{} relations with most arcs".format(n_most))
    logger.info("*"*50)
    logger.info("{}".format(most_common_relations[:n_most]))
    logger.info("*"*50)

    arc_counter = KG.get_arc_counter()
    pairs_more_than_one = [(arc, count) for arc, count in arc_counter.items() if count > 1]
    n = len(pairs_more_than_one)
    logger.info("#pairs with +1 arcs/#total arcs {0}/{1} {2:0.3f}%".format(
        n, len(KG.triples), n/len(KG.triples)))

    duplicated_counter = KG.get_duplicated_counter()
    duplicated_more_than_one = [(arc, count) for arc, count in duplicated_counter.items() if count > 1]
    n_duplicated = len(duplicated_more_than_one)
    logger.info("#duplicated pairs with +1 arcs/#total arcs {0}/{1} {2:0.3f}%".format(
        n_duplicated, len(KG.triples), n_duplicated/len(KG.triples)*100))
    logger.info('#true multirelational pairs {}/{} {:.3f}'.format(
        n-n_duplicated, len(KG.triples), n-n_duplicated/len(KG.triples)*100))

    if n > 0:
        logger.info("{} most common pairs with more than one arc".format(n_most))
        logger.info("*"*50)
        logger.info(arc_counter.most_common(n_most))
        logger.info("*"*50)



def stats_relations(KG):
    """Print out stats on each relation"""
    adjacencies = KG.get_adjacencies()
    arcs_per_relation = KG.get_arcs_per_relation()
    n_entities = len(KG.entities)

    results = []

    for rel in adjacencies:
        logger.info("stats for relation -- {}".format(rel))
        n_sources, n_targets = len(KG.rel2ents[rel]['sources']), len(KG.rel2ents[rel]['targets'])
        logger.info("# sources: {}, # targets: {}".format(n_sources, n_targets))

        n_arcs = arcs_per_relation[rel]
        neg_potential_full = n_entities * (n_entities - 1) - n_arcs
        neg_bipartite_directed = n_sources * n_targets - n_arcs
        neg_bipartite_undirected = 2 * n_sources * n_targets - n_arcs

        logger.info("# arcs: {}".format(n_arcs))
        logger.info("positive to negative ratio:")
        logger.info("bipartite directed 1:{:.2f}, bipartite undirected 1:{:.2f}, full 1:{:.2f}".format(
            neg_bipartite_directed/n_arcs, 
            neg_bipartite_undirected/n_arcs, 
            neg_potential_full/n_arcs))

        stat_object = {
            'relation': rel,
            'n_arcs': n_arcs,
            'sources': n_sources,
            'targets': n_targets,
            'biparted_directed': neg_bipartite_directed/n_arcs,
            'biparted_undirected': neg_bipartite_undirected/n_arcs, 
            'full': neg_potential_full/n_arcs,
        }

        results.append(stat_object)

    return results


@click.command()
@click.argument("graph-path", type=click.Path(exists=True))
@click.option("--out-file", default='./stats/stats_relations.csv')
@click.option("--dry-run", default=False, is_flag=True)
def main(graph_path, out_file, dry_run):
    logger.info("reading graph from {}".format(graph_path))
    KG = init_graph(graph_path)
    logger.info(KG)
    stats_graph(KG)
    results = stats_relations(KG)

    if not dry_run:
        logger.info("writing relation stats to {}".format(out_file))
        folder = os.path.dirname(out_file)
        if not os.path.exists(folder):
            logger.info("creating {} (did not exist)".format(folder))
            os.makedirs(folder)

        df = pd.DataFrame(results)
        df.to_csv(out_file, index=None)


if __name__ == "__main__":
    main()
