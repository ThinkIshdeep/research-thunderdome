import base64, io, json, math, unittest
import numpy as np
from lab.core import astar, clean
from lab import automl

class CoreTests(unittest.TestCase):

    def test_astar_routes_around_wall(self):
        path = astar([0, 0], [4, 4], [[2, y] for y in range(4)], 5)
        self.assertEqual(path[-1], [4, 4])
        self.assertTrue(all((p[0] != 2 or p[1] == 4 for p in path)))

    def test_astar_impossible(self):
        self.assertEqual(astar([0, 0], [2, 2], [[1, y] for y in range(3)], 3), [])

    def test_json_serialization(self):
        self.assertEqual(clean({'a': np.int64(2), 'b': float('nan')}), {'a': 2, 'b': None})

class ExperimentTests(unittest.TestCase):

    def test_runner_rejects_code(self):
        self.assertFalse(automl.Engine().audit({'issue': 'arbitrary_code'})['accepted'])

    def test_disjoint_splits(self):
        e = automl.Engine()
        r = e.campaign({'trials': 3, 'dataset': 'wine'})
        a, b, c = [set(r['split'][k]) for k in ['train', 'validation', 'test']]
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(len(r['trials']), 3)
