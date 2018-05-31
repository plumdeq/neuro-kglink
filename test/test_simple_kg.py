# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Start with a simple KG

e1 r1 e2
e1 r2 e2
e3 r2 e4
e4 r3 e5

we want to have the following:

* [x] entities = { e1, e2, e3, e4, e5 }
* [x] relations = { r1, r2, r3 }
* [x] for both entites and relations we should have ent2idx and idx2ent mappings 
  *-> idx2ent[ent2idx[e1]] == e1
* [x] for all relations we need to have adjacencies A_ri : |entities| x |entities|, 
  A_ri(i, j) = 1 if idx2ent[i] ri idx2ent[j] \in relations, and
  A_ri(i, j) = 0 if idx2ent[i] ri idx2ent[j] \not \in relations
* [x] for all A_ri we should have inverted adjacencies A'_ri(i, j) = 1 if A_ri(i, j) == 0, and vice versa
* [x] generate negatives for e1 with A_ri[ent2idx[e1], :] == 0 and 
  for e2 with A_ri[:, ent2idx[e2]] == 0
* [x] generate data loader with indicator function, e1, rel, [1, 0, ..., 1]
  (according to the adjacencies)

"""
# Standard-library imports
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# sys.path.insert(0, path)

# Third-party imports
import numpy as np
import pytest


def test_entities_and_relations(sample_kg):
    obj = sample_kg
    KG = obj['KG']

    assert set(KG.entities) == set(obj["entities"])
    assert set(KG.relations) == set(obj["relations"])


def test_mappings(sample_kg):
    obj = sample_kg
    KG = obj['KG']

    for (e, r) in zip(obj["entities"], obj["relations"]):
        assert KG.idx2ent[KG.ent2idx[e]] == e
        assert not r in KG.ent2idx

    for (e, r) in zip(obj["entities"], obj["relations"]):
        assert KG.idx2rel[KG.rel2idx[r]] == r
        assert not e in KG.rel2idx

    logger.info("All relation indices must be higher than entities")
    rel_indices = KG.rel2idx.values()
    for e in obj["entities"]:
        e_ix = KG.ent2idx[e]
        assert all(e_ix < rel_ix for rel_ix in rel_indices)

def test_relation2triples(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    triples = obj['triples']

    for i, (e1, r, e2) in enumerate(triples):
        assert i in KG.rel2triples[r]


def test_relation2entities(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx

    assert set([M['e1'], M['e3']]) == set(KG.rel2ents['r1']['sources'])
    assert set([M['e2']]) == set(KG.rel2ents['r1']['targets'])
    assert set([M['e1'], M['e3']]) == set(KG.rel2ents['r2']['sources'])
    assert set([M['e2'], M['e4']]) == set(KG.rel2ents['r2']['targets'])
    assert set([M['e4']]) == set(KG.rel2ents['r3']['sources'])
    assert set([M['e5']]) == set(KG.rel2ents['r3']['targets'])


def test_adjacencies_all_triples(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    n_entities = len(obj["entities"])

    # get adjacencies taking into account all triples
    A = KG.get_adjacencies()

    # assert (M['e1'], M['e1']) in A["r1"]["pairs"]
    assert (M['e1'], M['e2']) in A["r1"]["pairs"]
    assert (M['e2'], M['e1']) not in A["r1"]["pairs"] # directed
    assert (M['e3'], M['e2']) in A["r1"]["pairs"]
    assert (M['e2'], M['e3']) not in A["r1"]["pairs"]
    assert (M['e4'], M['e3']) not in A["r1"]["pairs"]
                                          
    assert (M['e1'], M['e2']) in A["r2"]["pairs"]
    assert (M['e3'], M['e4']) in A["r2"]["pairs"]
    assert (M['e2'], M['e3']) not in A["r2"]["pairs"]
                                         
    assert (M['e4'], M['e5']) in A["r3"]["pairs"]
    assert (M['e5'], M['e4']) not in A["r3"]["pairs"]
    assert (M['e2'], M['e3']) not in A["r3"]["pairs"]


def test_adjacencies_train_triples(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    n_entities = len(obj["entities"])
    train_triples = obj["train_triples"]

    # get adjacencies taking into account all triples
    A = KG.get_adjacencies(triples=train_triples)

    tr_e1s, tr_rels, tr_e2s = zip(*train_triples)
    neg_e1s = [e for e in KG.entities if e not in tr_e1s]
    neg_rels = [e for e in KG.entities if e not in tr_rels]
    neg_e2s = [e for e in KG.entities if e not in tr_e2s]

    for e1, rel, e2 in train_triples:
        assert (M[e1], M[e2]) in A[rel]["pairs"]

        for neg_e1 in neg_e1s:
            if neg_e1 != e2:
                assert (M[neg_e1], M[e2]) not in A[rel]["pairs"]

        for neg_e2 in neg_e2s:
            if e1 != neg_e2:
                assert (M[e1], M[neg_e2]) not in A[rel]["pairs"]


def test_negatives_idx(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    A = KG.get_adjacencies()

    assert set(KG.negatives(A, 'e1', 'r1', None)) == set([M['e3'], M['e4'], M['e5']])
    assert set(KG.negatives(A, None, 'r1', 'e2')) == set([M['e4'], M['e5']])
    assert set(KG.negatives(A, 'e1', 'r2', None)) == set([M['e3'], M['e4'], M['e5']])
    assert set(KG.negatives(A, 'e4', 'r3', None)) == set([M['e1'], M['e2'], M['e3']])


def test_negatives_ent(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    A = KG.get_adjacencies()

    assert set(KG.negatives(A, 'e1', 'r1', None, internal_idx=False)) == set(['e3', 'e4', 'e5'])
    assert set(KG.negatives(A, None, 'r1', 'e2', internal_idx=False)) == set(['e4', 'e5'])
    assert set(KG.negatives(A, 'e1', 'r2', None, internal_idx=False)) == set(['e3', 'e4', 'e5'])
    assert set(KG.negatives(A, 'e4', 'r3', None, internal_idx=False)) == set(['e1', 'e2', 'e3'])


def test_bulk_sample_positives_sanity(sample_kg):
    logger.info("should sample at most N samples")

    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    A = KG.get_adjacencies()

    for rel in KG.relations:
        n_samples = A[rel]['A'].count_nonzero() # max number of arcs
        sources, targets = KG.bulk_sample_positives(A[rel]['A'])
        assert len(sources) == len(targets) == n_samples

        all_inside(A[rel]['pairs'], sources, targets)

        # asking for more samples will give the max
        sources, targets = KG.bulk_sample_positives(A[rel]['A'], n_samples=n_samples+100)
        assert len(sources) == len(targets) == n_samples

        all_inside(A[rel]['pairs'], sources, targets)

        # asking for fewer samples gives that number
        sources, targets = KG.bulk_sample_positives(A[rel]['A'], n_samples=1)
        assert len(sources) == len(targets) == 1

        all_inside(A[rel]['pairs'], sources, targets)


def test_bulk_sample_negatives_sanity(sample_kg):
    logger.info("should sample at most N samples")

    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    A = KG.get_adjacencies()

    n_ents = len(KG.entities)

    for rel in KG.relations:
        n_samples = n_ents * (n_ents-1) - A[rel]['A'].count_nonzero() # max number of arcs
        negs = list(KG.bulk_sample_negatives_generator(A[rel]['A']))
        assert len(negs) == n_samples

        all_outside(A[rel]['pairs'], negs)

        # asking for more samples will give the max
        negs = list(KG.bulk_sample_negatives_generator(A[rel]['A'], n_samples=n_samples+100))
        assert len(negs) == n_samples

        all_outside(A[rel]['pairs'], negs)

        # asking for fewer samples gives that number
        negs = list(KG.bulk_sample_negatives_generator(A[rel]['A'], n_samples=1))
        assert len(negs) == 1

        all_outside(A[rel]['pairs'], negs)


def test_bulk_sample_fixed_sources_targets(sample_kg):
    obj = sample_kg
    KG = obj['KG']
    M = KG.ent2idx
    A = KG.get_adjacencies()

    for rel in KG.relations:
        sources, targets = KG.rel2ents[rel]["sources"], KG.rel2ents[rel]["targets"]
        n_positives = A[rel]['A'].count_nonzero()
        n_samples_directed = len(sources) * len(targets) - n_positives
        n_samples_undirected = (2 * len(sources) * len(targets)) - n_positives

        negs_directed = list(KG.bulk_sample_negatives_generator(A[rel]['A'],
                             sources=sources, targets=targets))
        negs_undirected = list(KG.bulk_sample_negatives_generator(A[rel]['A'],
                               sources=sources, targets=targets, directed=False))

        assert len(negs_directed) == n_samples_directed
        assert len(negs_undirected) == n_samples_undirected

        all_outside(A[rel]['pairs'], negs_directed)
        all_outside(A[rel]['pairs'], negs_undirected)

# assume that A[rel]['pairs'] was correctly constructed
def all_inside(pairs, sources, targets):
    for s, t in zip(sources, targets):
        assert (s, t) in pairs


# assume that A[rel]['pairs'] was correctly constructed
def all_outside(pairs, neg_generator):
    for s, t in neg_generator:
        assert (s, t) not in pairs
