# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

General KG class, prepares entities, relations, adjacency and inverted
adjacency matrices for each relation, and keeps hashes for entities and
relations.

This is a general callable which can be used for various tasks:

* folds generation
* evaluation metrics of link prediction and full reconstruction (as in IR)

"""
# # Standard-library imports
import time
import random
import logging
import itertools as it
from collections import Counter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Third-party imports
import numpy as np
import scipy as sp
import scipy.sparse
import click


class KG(object):
    """
    We assume that the triples are in ei, rk, ej order, if source and objects
    are different then provide their indices

    """
    def __init__(self, triples, e1_index=0, rel_index=1, e2_index=2):
        self.triples = triples
        self.e1_index = e1_index
        self.rel_index = rel_index
        self.e2_index = e2_index
        self.entities = self.get_entities()
        self.relations = self.get_relations()

        # different mappings, dictionaries
        self.ent2idx, self.idx2ent = self.get_entity_mappings()
        self.rel2idx, self.idx2rel = self.get_relation_mappings()
        self.rel2triples = self.get_rel2triples_mappings()
        self.rel2ents = self.get_rel2ents_mappings()


    def get_entities(self, triples=None, e1_index=None, e2_index=None):
        """List of all unique entities (subject and objects)"""
        triples = triples or self.triples
        e1_index = self.e1_index
        e2_index = self.e2_index

        entities = [l[e1_index] for l in triples] +\
                   [l[e2_index] for l in triples]

        return list(set(entities))


    def get_relations(self, triples=None, rel_index=None):
        """List of all unique relations (predicates)"""
        triples = triples or self.triples
        rel_index = self.rel_index

        relations = [l[rel_index] for l in triples]

        return list(set(relations))


    def get_entity_mappings(self):
        """entity -> idx, and its inverse"""
        ent2idx = { e:i for i, e in enumerate(self.entities) }
        idx2ent = { i:e for e, i in ent2idx.items() }

        return ent2idx, idx2ent


    def get_relation_mappings(self):
        """relation -> idx, and its inverse, first we need to have the ent2idx
        map"""
        if not self.entities:
            raise Exception("We need to know how many entities are there")

        n_entities = len(self.entities)

        rel2idx = { r:i for i, r in enumerate(self.relations, n_entities) }
        idx2rel = { i:r for r, i in rel2idx.items() }

        return rel2idx, idx2rel


    def get_rel2triples_mappings(self):
        """Maintains a mapping of relation -> triple indices"""
        rel2triples = {}

        for rel in self.relations:
            rel2triples[rel] = [i for i, l in enumerate(self.triples) if l[self.rel_index] == rel]

        return rel2triples


    def get_rel2ents_mappings(self):
        """Maintains a mapping of relation -> sources indices and target indices"""
        rel2ents = {}

        for rel in self.relations:
            rel2ents[rel] = {}
            rel2ents[rel]['sources'] = []
            rel2ents[rel]['targets'] = []

            for l in self.triples:
                if l[self.rel_index] == rel:
                    e1, e2 = self.ent2idx[l[self.e1_index]], self.ent2idx[l[self.e2_index]]
                    rel2ents[rel]['sources'].append(e1)
                    rel2ents[rel]['targets'].append(e2)

            # we only need unique sources and targets 
            rel2ents[rel]['sources'] = list(set(rel2ents[rel]['sources']))
            rel2ents[rel]['targets'] = list(set(rel2ents[rel]['targets']))

        return rel2ents


    def get_adjacencies(self, triples=None):
        """Builds adjacencies according to the known triples gives in `triples`"""
        triples = triples or self.triples
        A = {}
        M = self.ent2idx # we only store indices for efficiency
        n_entities = len(self.entities)

        for rel in self.relations:
            A[rel] = {}
            A[rel]["pairs"] = []

            for l in triples:
                if l[self.rel_index] == rel:
                    A[rel]["pairs"].append((M[l[self.e1_index]], M[l[self.e2_index]]))

            e1_idxs, e2_idxs = [], []

            if A[rel]["pairs"] != []:
                e1_idxs, e2_idxs = zip(*A[rel]["pairs"])

            assert len(e1_idxs) == len(e2_idxs)

            I, J = np.array(e1_idxs), np.array(e2_idxs)

            A[rel]["A"] = sp.sparse.csr_matrix(
                    (np.ones(len(e1_idxs)), (I, J)), 
                    shape=(n_entities, n_entities), dtype=np.int32)

        return A


    def get_adjacency(self, rel, triples=None):
        """Builds adjacency matrix for relation `rel` according to the known triples gives in `triples`"""
        triples = triples or self.triples
        A = {}
        M = self.ent2idx # we only store indices for efficiency
        n_entities = len(self.entities)

        A["pairs"] = []

        for l in triples:
            if l[self.rel_index] == rel:
                A["pairs"].append((M[l[self.e1_index]], M[l[self.e2_index]]))

        e1_idxs, e2_idxs = [], []

        if A["pairs"] != []:
            e1_idxs, e2_idxs = zip(*A["pairs"])

        assert len(e1_idxs) == len(e2_idxs)

        I, J = np.array(e1_idxs), np.array(e2_idxs)

        A["A"] = sp.sparse.csr_matrix(
                (np.ones(len(e1_idxs)), (I, J)), 
                shape=(n_entities, n_entities), dtype=np.int32)

        return A


    def adjacency_row(self, A, e1=None, rel=None, e2=None):
        """Extracts adjacency row from the adjacency matrix"""
        if (not e1 is None) and (not e2 is None):
            raise Exception("one of e1 or e2 should be None")

        if e1 is None and e2 is None:
            raise Exception("both e1 and e2 cannot be None")

        M = self.ent2idx
        M_inv = self.idx2ent

        indicator = None

        if e2 is None:
            indicator = A.getrow(M[e1]).toarray().reshape(-1)

        if e1 is None:
            indicator = A.getcol(M[e2]).toarray().reshape(-1)

        return indicator


    def negatives(self, A, e1=None, rel=None, e2=None, internal_idx=True):
        """
        Give all negatives for the left or right entity and the relation

        """
        A_rel = A[rel]["A"]
        M = self.ent2idx
        M_inv = self.idx2ent

        indicator = self.adjacency_row(A_rel, e1=e1, rel=rel, e2=e2)

        indices = np.where(indicator == 0)[0].tolist()

        # make sure we don't have self loops
        if e1:
            indices = [i for i in indices if not i == M[e1]]
        elif e2:
            indices = [i for i in indices if not i == M[e2]]
        else:
            raise Exception("Cannot happen either e1 or e2 should be None")

        if internal_idx:
            return indices
        else:
            return [M_inv[idx] for idx in indices]


    def negatives_rel(self, A_rel, e1=None, rel=None, e2=None, internal_idx=True):
        """
        Give all negatives for the fixed relation, we need the explicit
        adjacency matrix

        """
        M = self.ent2idx
        M_inv = self.idx2ent

        indicator = self.adjacency_row(A_rel, e1=e1, rel=rel, e2=e2)

        indices = np.where(indicator == 0)[0].tolist()

        # make sure we don't have self loops
        if e1:
            indices = [i for i in indices if not i == M[e1]]
        elif e2:
            indices = [i for i in indices if not i == M[e2]]
        else:
            raise Exception("Cannot happen either e1 or e2 should be None")

        if internal_idx:
            return indices
        else:
            return [M_inv[idx] for idx in indices]



    def positives(self, A, e1=None, rel=None, e2=None, internal_idx=True):
        """
        Give all positives for the left or right entity and the relation

        """
        A_rel = A[rel]["A"]
        M = self.ent2idx
        M_inv = self.idx2ent

        indicator = self.adjacency_row(A_rel, e1=e1, rel=rel, e2=e2)

        indices = np.where(indicator == 1)[0].tolist()
        
        # make sure we don't have self loops
        if e1:
            indices = [i for i in indices if not i == M[e1]]
        elif e2:
            indices = [i for i in indices if not i == M[e2]]
        else:
            raise Exception("Cannot happen either e1 or e2 should be None")

        if internal_idx:
            return indices
        else:
            return [M_inv[idx] for idx in indices]


    def bulk_sample_positives(self, A, n_samples=None, shuffle=True):
        """Sample n positives from the adjacency matrix A, if n is None, then
        return all samples. If we are given the possible sources and targets
        then we will restrict our samples to these nodes"""
        sources, targets = A.nonzero()
        N = len(sources)

        if n_samples is None:
            logger.info("Bulk sample all possible pairs {}".format(N))
            n_samples = N
        elif isinstance(n_samples, int) and n_samples > N:
            logger.info("There are less possible pairs than requested number of samples ({} < {})".format(N, n_samples))
            n_samples = N
        else:
            logger.info("Will sample {} from {}".format(n_samples, N))

        ixs = list(range(N))
        if shuffle:
            random.shuffle(ixs)

        sampled_ixs = ixs[:n_samples]

        return sources[sampled_ixs], targets[sampled_ixs]


    def bulk_sample_negatives_generator(self, A, n_samples=None, sources=None, targets=None, shuffle=True, directed=True):
        """Sample n negatives from the adjacency matrix A, if n is None, then
        return all samples. If we are given the possible sources and targets
        then we will restrict our samples to these nodes. If directed is True
        we will preserve the directedness"""

        if sources is not None and targets is not None:
            sources, targets = np.array(sources), np.array(targets)
        else:
            n_ents = len(self.entities)
            sources, targets = np.array(range(n_ents)), np.array(range(n_ents))
            logger.info("Samples are not restricted to sources and targets")

        if shuffle:
            logger.info("shuffling sources and targets")
            source_ixs = list(range(len(sources)))
            target_ixs = list(range(len(targets)))
            random.shuffle(source_ixs)
            random.shuffle(target_ixs)
            sources, targets = sources[source_ixs], targets[target_ixs]

        logger.info("Generating negatives")
        counter = 0

        cartesian_product = it.product(sources, targets)
        if not directed:
            logger.info("undirected bipartite graph negative generation")
            cartesian_product = it.chain(it.product(sources, targets),
                                         it.product(targets, sources))

        for neg_s, neg_t in cartesian_product:
            if counter == n_samples:
                raise StopIteration

            if neg_s == neg_t:
                # do not generate self loops
                continue

            if A[neg_s,neg_t] == 0:
                counter += 1
                yield neg_s, neg_t


    def get_arc_counter(self, triples=None):
        """
        Counts the number of arcs between any pair of nodes, reports if any pair of
        nodes has more than one arc. This can be computed for a specific set of
        triples, or for all known triples

        """
        triples = triples or self.triples

        arcs = [(l[self.e1_index], l[self.e2_index]) for l in triples]
        arc_counter = Counter(arcs)
        
        return arc_counter


    def get_duplicated_counter(self, triples=None):
        """
        Check number of duplicated triples in the list

        """
        triples = triples or self.triples

        arcs = [(l[self.e1_index], l[self.rel_index], l[self.e2_index]) for l in triples]
        duplicated_counter = Counter(arcs)
        
        return duplicated_counter


    def get_arcs_per_relation(self, triples=None):
        """Computes number of arcs that each relation has"""
        A = self.get_adjacencies(triples)

        return { rel: A[rel]['A'].count_nonzero() for rel in self.relations }


    def __repr__(self):
        fmt_string = "KG contains:\n"
        fmt_string += "Num entities: {}, num relations {}".format(
            len(self.entities), len(self.relations))

        return fmt_string
