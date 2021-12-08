"""
Unit tests for generator class
"""

import pandas as pd

from . import get_labels_from_csv

def test_get_labels_from_csv_shape():
    data = get_labels_from_csv('data.csv')
    assert data.shape == (355, 28)
