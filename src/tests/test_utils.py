import numpy as np

from datetime import datetime
from unittest import TestCase

from ..utils import read_roti, load_data, read_gfz

class TestUtils(TestCase):

    def test_read_roti(self):
        date, lats, rows = read_roti("/content/roti/data/roti0010.10f")
        self.assertEqual(date, datetime(2010, 1, 1))
        self.assertEqual(lats.tolist(), list(range(89, 50, -2)))
        self.assertEqual(rows.shape, (20, 180))

        date, lats, rows = read_roti("/content/roti/data/roti1970.13f")
        self.assertEqual(date, datetime(2013, 7, 16))
        self.assertEqual(lats.tolist(), list(range(89, 50, -2)))
        self.assertEqual(rows.shape, (20, 180))

        date, lats, rows = read_roti("/content/roti/data/roti1920.20f")
        self.assertEqual(date, datetime(2020, 7, 10))
        self.assertEqual(lats.tolist(), list(range(89, 50, -2)))
        self.assertEqual(rows.shape, (20, 180))

    def test_load_data(self):
        data = load_data(datetime(2010, 1, 1), datetime(2010, 1, 3))
        self.assertEqual(data.shape, (2, 26))
        self.assertEqual(list(data.index), [datetime(2010, 1, 1), datetime(2010, 1, 2)])

        self.assertRaises(ValueError, load_data, datetime(2010, 1, 3), datetime(2010, 1, 1))

    def test_read_gfz(self):
        data = read_gfz()
        data_2010_01_01 = data.loc[datetime(2010, 1, 1)]
        self.assertEqual(data_2010_01_01["F10.7obs"], 75.2)
        data_2013_01_25 = data.loc[datetime(2013, 1, 25)]
        self.assertEqual(data_2013_01_25["D"], 2.0)
        data_2020_11_19 = data.loc[datetime(2020, 11, 19)]
        self.assertEqual(data_2020_11_19["days"], 32465.0)
