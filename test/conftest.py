# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Fixtures shared for the test suits

"""
# Standard-library imports
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, path)

# Third-party imports
import pytest

# Cross-library imports
from kglink import kg 


@pytest.fixture(scope="function")
def sample_kg():
    triples = [
        ['e1', 'r1', 'e2'],
        ['e3', 'r1', 'e2'],
        ['e1', 'r2', 'e2'],
        ['e3', 'r2', 'e4'],
        ['e4', 'r3', 'e5'],
    ]

    train_triples = triples[:3]
    val_triples = [triples[3]]
    test_triples = [triples[-1]]

    entities = ['e1', 'e2', 'e3', 'e4', 'e5']
    relations = ['r1', 'r2', 'r3']

    KG = kg.KG(triples)

    return {
        'triples': triples,
        'entities': entities,
        'relations': relations,
        'KG': KG,
        'train_triples': train_triples,
        'val_triples': val_triples,
        'test_triples': test_triples,
    }
