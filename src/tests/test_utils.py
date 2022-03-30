import numpy as np

from datetime import datetime
from unittest import TestCase

from ..utils import read_roti, load_data, get_date_from_filename

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
        keys = data.keys()
        self.assertEqual(len(keys), 2)
        self.assertEqual(list(keys), [datetime(2010, 1, 1), datetime(2010, 1, 2)])

    def test_get_date_from_filename(self):
        date = get_date_from_filename("roti0010.10f")
        self.assertEqual(date, datetime(2010, 1, 1))

        date = get_date_from_filename("roti1970.13f")
        self.assertEqual(date, datetime(2013, 7, 16))

        date = get_date_from_filename("roti1920.20f")
        self.assertEqual(date, datetime(2020, 7, 10))
