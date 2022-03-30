import numpy as np

from datetime import datetime
from unittest import TestCase

from ..utils import read_roti

class TestUtils(TestCase):

    def test_read_roti(self):
        date, lats, rows = read_roti("/content/roti/data/roti0010.10f")
        self.assertEqual(date, datetime(2010, 1, 1))
        self.assertEqual(lats.tolist(), list(range(89, 50, -2)))
