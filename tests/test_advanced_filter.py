#!/usr/bin/env python3
"""Unit tests for pyruvector advanced filtering"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../target/release"))

from pyruvector import PayloadIndexManager, FilterBuilder, FilterEvaluator, IndexType


class TestBasicFiltering(unittest.TestCase):
    def setUp(self):
        self.manager = PayloadIndexManager()

    def test_create_index(self):
        self.manager.create_index("category", IndexType.Keyword)
        indices = self.manager.list_indices()
        self.assertIn("category", indices)

    def test_filter_evaluation(self):
        self.manager.create_index("price", IndexType.Float)
        self.manager.index_payload("p1", {"price": 100.0})
        
        evaluator = FilterEvaluator(self.manager)
        filter_dict = FilterBuilder().eq("price", 100.0).build()
        results = evaluator.evaluate(filter_dict)
        self.assertIn("p1", results)


if __name__ == "__main__":
    unittest.main()
