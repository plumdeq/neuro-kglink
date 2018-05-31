# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Link prediction data generation

* all positives arcs and all negatives arcs

## Implementation steps

### Single label classification

1. Generate retained graph - (full KG - alpha of specific relation)
    * alpha 10% or 20% etc.
2. Train and test splits for classification of single labels

### Multi-label classification

1. Retained graph where each relation instances are deleted with a divided
ratio
2. Train and test splits for multi-label classification

"""
# Standard-library imports
import os
import sys
import logging
import math
import random

# Third-party imports
import click
import numpy as np
from tqdm import tqdm
import pandas as pd

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, path)

# Cross-library imports
from kglink import kg, dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOEHNDORF_GRAPH = "/media/Warehouse/bigdata-muw/bio-kg/hoehndorf/hoehndorf-graph-improved/merged-graphs/hierarchy_path/data/hierarchy.nt"
EXCLUDE_RELS = [
        "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
        "<http://www.w3.org/2000/01/rdf-schema#subClassOf>",
        ]


def triple_output_format(KG, triple, output_fmt='pair', internal_idx=True):
    """Format triple when we write to files

    * if output_fmt == 'pair':
        * if internal_idx:
            * (s, p, o) -> s_ix, p_ix
        * else
            * (s, p, o) -> s, p
    * else:
        * if internal_idx:
            * (s, p, o) -> s_ix, p_ix, o_ix
        * else
            * (s, p, o) -> s, p, o

    """
    if output_fmt == 'pair':
        if internal_idx:
            return "{} {}\n".format(
                    KG.ent2idx[triple[KG.e1_index]], KG.ent2idx[triple[KG.e2_index]])
        else:
            return "{} {}\n".format(
                    triple[KG.e1_index], triple[KG.e2_index])
    else:
        if internal_idx:
            return "{} {} {}\n".format(
                    KG.ent2idx[triple[KG.e1_index]],
                    KG.rel2idx[triple[KG.rel_index]], 
                    KG.ent2idx[triple[KG.e2_index]])
        else:
            return "{} {} {}\n".format(
                    triple[KG.e1_index], triple[KG.rel_index], triple[KG.e2_index])


def init_graph(path):
    """Read lines from the graph and initialize the KG object"""
    lines = None
    with open(path, "r") as f:
        lines = [l.strip().split()[:3] for l in f.readlines()]

    KG = kg.KG(lines)

    return KG


def retain_arcs_on_relation(KG, rel, ratio=0.9):
    """Retain edges for the given ratio"""
    triple_ixs = KG.rel2triples[rel]
    logger.info("Shuffling {} triples".format(len(triple_ixs)))
    random.shuffle(triple_ixs)
    threshold = math.floor(len(triple_ixs)*ratio)

    logger.info("retaining {} triples".format(threshold))
    retained_ixs, unretained_ixs = triple_ixs[:threshold], triple_ixs[threshold:]
    # NOTE that unretained WILL CONTAIN test positives!

    return retained_ixs, unretained_ixs

def write_triples(KG, unretained_ixs, retained_path, unretained_path):
    """Write triples corresponding to retained and unretained graphs"""
    # retained_path, unretained_path = graph_paths['train_graph'], graph_paths['test_graph']

    logger.info("identifying triples to retain")
    all_ixs = np.array(range(len(KG.triples)))
    retained_mask = np.ones(len(KG.triples), dtype=bool)
    retained_mask[unretained_ixs] = False

    # retained_ixs = (i for i in range(len(KG.triples)) if i not in unretained_ixs)
    retained_ixs = all_ixs[retained_mask]

    logger.info("writing train graph to {}".format(retained_path))
    with open(retained_path, 'w') as f:
        f.writelines(triple_output_format(KG, KG.triples[i]) for i in retained_ixs)

    logger.info("writing test graph to {}".format(unretained_path))
    with open(unretained_path, 'w') as f:
        for ix in unretained_ixs:
            l = KG.triples[ix]
            f.write(triple_output_format(KG, l))


def write_single_label_tasks(KG, graph_paths, retained_ixs, unretained_ixs, biased_negatives, unbiased_negatives, ratio):
    """Write data for single label training and testing with biased and
    unbiased negatives""" 
    n_neg_biased = len(biased_negatives)
    n_neg_unbiased = len(unbiased_negatives)

    # extract pairs of nodes from the triples
    tr_positives = [(KG.ent2idx[KG.triples[i][KG.e1_index]], KG.ent2idx[KG.triples[i][KG.e2_index]]) for i in retained_ixs]
    te_positives = [(KG.ent2idx[KG.triples[i][KG.e1_index]], KG.ent2idx[KG.triples[i][KG.e2_index]]) for i in unretained_ixs]

    overlap_tr_pos = len(set(tr_positives) & set(te_positives))
    logger.info("Overlap train positives and test positives {}".format(overlap_tr_pos))
    
    neg_biased_ratio = math.floor(n_neg_biased*ratio)
    tr_neg_biased, te_neg_biased = biased_negatives[:neg_biased_ratio], biased_negatives[neg_biased_ratio:]
    overlap_tr_neg_biased = len(set(tr_neg_biased) & set(te_neg_biased))
    logger.info("Overlap train and test negatives biased {}".format(overlap_tr_neg_biased))

    neg_unbiased_ratio = math.floor(n_neg_unbiased*ratio)
    tr_neg_unbiased, te_neg_unbiased = unbiased_negatives[:neg_biased_ratio], unbiased_negatives[neg_biased_ratio:]
    overlap_tr_neg_unbiased = len(set(tr_neg_unbiased) & set(te_neg_unbiased))
    logger.info("Overlap train and test negatives unbiased {}".format(overlap_tr_neg_unbiased))

    biased_overlap1 = len(set(tr_neg_biased) & set(te_positives))
    biased_overlap2 = len(set(tr_neg_biased) & set(tr_positives))
    logger.info("Overlap train biased negatives and test positives {}".format(biased_overlap1))
    logger.info("Overlap train biased negatives and train positives {}".format(biased_overlap2))

    unbiased_overlap1 = len(set(tr_neg_unbiased) & set(te_positives))
    unbiased_overlap2 = len(set(tr_neg_unbiased) & set(tr_positives))
    logger.info("Overlap train unbiased negatives and test positives {}".format(unbiased_overlap1))
    logger.info("Overlap train unbiased negatives and train positives {}".format(unbiased_overlap2))


    tr_biased_path, te_biased_path = graph_paths['train_biased_single_label'], graph_paths['test_biased_single_label']
    tr_unbiased_path, te_unbiased_path = graph_paths['train_unbiased_single_label'], graph_paths['test_unbiased_single_label']

    logger.info("Writing biased single label prediction task to {} and {}".format(
        tr_biased_path, te_biased_path))

    with open(tr_biased_path, 'w') as f:
        f.writelines("{} {} 1\n".format(l[0], l[1]) for l in tr_positives)
        f.writelines("{} {} 0\n".format(l[0], l[1]) for l in tr_neg_biased)

    with open(te_biased_path, 'w') as f:
        f.writelines("{} {} 1\n".format(l[0], l[1]) for l in te_positives)
        f.writelines("{} {} 0\n".format(l[0], l[1]) for l in te_neg_biased)

    logger.info("Writing unbiased single label prediction task to {} and {}".format(
        tr_unbiased_path, te_unbiased_path))

    with open(tr_unbiased_path, 'w') as f:
        f.writelines("{} {} 1\n".format(l[0], l[1]) for l in tr_positives)
        f.writelines("{} {} 0\n".format(l[0], l[1]) for l in tr_neg_unbiased)

    with open(te_unbiased_path, 'w') as f:
        f.writelines("{} {} 1\n".format(l[0], l[1]) for l in te_positives)
        f.writelines("{} {} 0\n".format(l[0], l[1]) for l in te_neg_unbiased)


    return {
        'overlap tr_pos and te_pos': overlap_tr_pos,
        'overlap tr_neg_biased and te_neg_biased': overlap_tr_neg_biased,
        'overlap tr_neg_unbiased and te_neg_unbiased': overlap_tr_neg_unbiased,
        'overlap tr_neg_biased and te_pos': biased_overlap1,
        'overlap tr_neg_biased and tr_pos': biased_overlap2,
        'overlap tr_neg_unbiased and te_pos': unbiased_overlap1,
        'overlap tr_neg_unbiased and tr_pos': unbiased_overlap2,
    }


def split_graph_link_prediction(KG, output_paths, ratio=0.9, 
                                exclude_rels=EXCLUDE_RELS,
                                retained_format='pairs',
                                dry_run=False,
                                folds=1, fold_start=1):
    """
    Read in KG in path split either on a specific relation (if given), or on
    all relations. Finally, prepare three datasets

    1. unsupervised - full KG - retained arcs (either on 1 or multiple relations)
    2. train - per relation 
    3. test - per relation

    If we are in the multi-relation case then prepare train and test splits per
    relation

    Args:
        output_paths (dict): relation -> path where we will write out results
        dry_run (bool): do not write out any files, only compute and report
            stats

    """
    stats = []
    for fold in tqdm(range(fold_start, folds + 1), desc="folds"):
        logger.info("fold is {}".format(fold))
        relations = [rel for rel in KG.relations if rel not in exclude_rels]
        all_unretained = []
        for rel in tqdm(relations, desc="relations"):
            logger.info("filtering triples for relation {}".format(rel)) 
            retained_ixs, unretained_ixs = retain_arcs_on_relation(KG, rel, ratio=ratio)
            all_unretained.extend(unretained_ixs)
            retained_triples = [KG.triples[i] for i in retained_ixs]
            unretained_triples = [KG.triples[i] for i in unretained_ixs]

            # retained is KG - retained arcs of a specific relation
            n_total = len(retained_ixs) + len(unretained_ixs) 

            A_rel_biased, A_rel_unbiased = KG.get_adjacency(rel), KG.get_adjacency(rel, triples=retained_triples)
            sources, targets = KG.rel2ents[rel]["sources"], KG.rel2ents[rel]["targets"]

            logger.info("generating biased negatives")
            negs_biased = list(KG.bulk_sample_negatives_generator(A_rel_biased['A'],
                               sources=sources, targets=targets,
                               n_samples=n_total))

            logger.info("generating unbiased negatives")
            negs_unbiased = list(KG.bulk_sample_negatives_generator(A_rel_unbiased['A'],
                                 sources=sources, targets=targets,
                                 n_samples=n_total))

            logger.info("#train positives {}".format(len(retained_ixs)))
            logger.info("#test positives {}".format(len(unretained_ixs)))
            logger.info("#biased negatives {}".format(len(negs_biased)))
            logger.info("#unbiased negatives {}".format(len(negs_unbiased)))

            n_samples_stats = {
                'n_tr_pos': len(retained_ixs),
                'n_te_pos': len(unretained_ixs),
                'n_neg_biased': len(negs_biased),
                'n_neg_unbiased': len(negs_unbiased),
            }

            logger.info("writing graphs and train/test splits in {}".format(output_paths[rel]))

            if not os.path.exists(output_paths[rel]):
                logger.info("creating directory {} (did not exist before)".format(output_paths[rel]))

                if not dry_run:
                    os.makedirs(output_paths[rel])

            graph_paths = {
                graph_type: os.path.join(
                    output_paths[rel], 
                    'relation-{}-{}-fold-{}'.format(KG.rel2idx[rel], graph_type, fold))
                for graph_type in ['train_graph', 'test_graph', 
                                   'train_biased_single_label', 'test_biased_single_label',
                                   'train_unbiased_single_label', 'test_unbiased_single_label',]
            }

            if not dry_run:
                write_triples(KG, unretained_ixs, graph_paths['train_graph'], 
                              graph_paths['test_graph'])
                stat_object = write_single_label_tasks(KG, graph_paths, retained_ixs, unretained_ixs, 
                                         negs_biased, negs_unbiased, ratio)
                stats.append(dict(fold=fold, relation=rel, **stat_object, **n_samples_stats))

        global_train_path = 'global-train-fold-{}'.format(fold)
        global_test_path = 'global-test-fold-{}'.format(fold)
        graph_paths['global_train'] = os.path.join(output_paths['global'], global_train_path)
        graph_paths['global_test'] = os.path.join(output_paths['global'], global_test_path)

        logging.info("writing global retained and unretained graphs")
        if not dry_run:
            write_triples(KG, all_unretained, graph_paths['global_train'], graph_paths['global_test'])

    return stats


def write_map_files(KG, output_path):
    """Write ent2idx and rel2idx to ids map"""
    relations_map_path = os.path.join(output_path, "rel2idx.map")
    logger.info("writing relations mapping id -> URI to {}".format(relations_map_path))

    with open(relations_map_path, 'w') as f:
        f.writelines("{}\t{}\n".format(value, key) for key, value in KG.rel2idx.items())

    entities_map_path = os.path.join(output_path, "ent2idx.map")
    logger.info("writing entities mapping id -> URI to {}".format(entities_map_path))

    with open(entities_map_path, 'w') as f:
        f.writelines("{}\t{}\n".format(value, key) for key, value in KG.ent2idx.items())


@click.command()
@click.option("--input-path", default=HOEHNDORF_GRAPH, type=click.Path(exists=True))
@click.option("--output-path", default="./output")
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--folds", default=1, type=int)
@click.option("--fold-start", default=1, type=int, help="number from which we start counting folds")
@click.option("--ratio", default=0.9, type=float, help="Ratio of train/test split")
@click.option("--stats-output", default="stats.csv")
def main(input_path, output_path, dry_run, folds, fold_start, ratio, stats_output):
    logger.info("reading graph from {}".format(input_path))
    logger.info("will generate train and test splits for {} folds with {} train_ratio".format(folds, ratio))
    logger.info("all output will be written in {}".format(output_path))
    logger.info("output format is 'rel-REL_ID-[train_graph|test_graph|train_single_label|test_single_label]-fold-FOLD_ID")
    if not os.path.exists(output_path):
        logger.info("creating directory {} (did not exist before)".format(output_path))
        os.makedirs(output_path)

    KG = init_graph(input_path)
    logger.info(KG)

    paths = { rel: os.path.join(output_path, str(KG.rel2idx[rel])) for rel in KG.relations }
    paths['global'] = output_path

    stats = split_graph_link_prediction(
            KG, paths, dry_run=dry_run, folds=folds, fold_start=fold_start, ratio=ratio)

    if not dry_run:
        write_map_files(KG, output_path)
        out_file = os.path.join(output_path, stats_output)
        logger.info("writing stats to {}".format(out_file))
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(out_file, index=None)


if __name__ == "__main__":
    main()
