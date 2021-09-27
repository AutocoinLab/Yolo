"""Tests for autopycoin loss functions."""

import numpy as np

import tensorflow as tf
from tensorflow import test
from tensorflow.python.keras import combinations
from . import model


@combinations.generate(combinations.combine(mode=["eager", "graph"]))
class ModelTest(test.TestCase):
    def test_config(self):
        assert True