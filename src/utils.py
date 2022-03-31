import os
import numpy as np

from datetime import datetime, timedelta

def read_roti(fpath):
    with open(fpath, 'rb') as f:
        header_read = False
        date = None
        lats = []
        rows = []
        line = f.readline()
        while not header_read or (line.strip() and not line.strip().startswith(b"END OF ROTIPOLARMAP")):
            if line.strip().startswith(b"START OF ROTIPOLARMAP"):
                header_read = True
            elif line.strip().startswith(b"END OF ROTIPOLARMAP") or line.strip().startswith(b"END OF FILE"):
                break
            elif not header_read:
                pass
            elif line[0:5].strip():
                date = datetime(int(line[0:7]), int(line[7:14]), int(line[14:21]))
            else:
                lat, lon_start, lon_end = float(line[3:9]), float(line[9:15]), float(line[15:21])
                lats.append(lat)
                block = [f.readline() for _ in range(18)]
                row = np.genfromtxt(block)
                rows.append(row.ravel())
            line = f.readline()
        return date, np.array(lats), np.array(rows)

def load_data(start_date = None, end_date = None):
    data = {}
    if start_date is None:
        start_date = datetime(2010, 1, 1)
    if end_date is None:
        end_date = datetime.now()
    for fname in sorted(os.listdir("/content/roti/data/")):
        date = get_date_from_filename(fname)
        if date >= start_date and date < end_date:
            d, lats, rows = read_roti("/content/roti/data/" + fname)
            assert date == d
            data[date] = rows
    return data

def get_date_from_filename(fname):
    doy = int(fname[4:7])
    year = 2000 + int(fname[9:11])
    return datetime(year, 1, 1) + timedelta(days=doy - 1)
