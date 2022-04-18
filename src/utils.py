import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

CONTENT_DIR = "/content/roti/"

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

def read_gfz():
    df = pd.read_csv(CONTENT_DIR + "GFZ.csv",
                     parse_dates={"date": ["YYYY", "MM", "DD"]},
                     date_parser=lambda y, m, d: datetime.strptime(y + m.zfill(2) + d.zfill(2), "%Y%m%d")
                     )
    df = df.set_index("date")
    return df

def load_data(start_date=None, end_date=None):
    '''
    start_date: datetime (included)
    end_date: datetime (not included)
    '''
    data = []
    if start_date is None:
        start_date = datetime(2010, 1, 1)
    if end_date is None:
        end_date = datetime.now()
    if start_date >= end_date:
        raise ValueError("Start date shoud be less than end date.")
    for fname in sorted(os.listdir(CONTENT_DIR + "data"), key=lambda x: (int(x[9:11]), int(x[4:8]))):
        date = get_date_from_filename(fname)
        if date >= end_date:
            break
        if date >= start_date:
            d, lats, rows = read_roti(CONTENT_DIR + "data" + "/" + fname)
            assert date == d
            data.append({"date": date, "ROTI": rows})
    df = pd.DataFrame.from_records(data, index="date")
    return df

def get_date_from_filename(fname):
    doy = int(fname[4:7])
    year = 2000 + int(fname[9:11])
    return datetime(year, 1, 1) + timedelta(days=doy - 1)

def plot_roti(data, date):
    plt.rcParams["font.size"] = 16
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    mlon = np.linspace(0, 360, 180)
    mlat = np.linspace(90, 50, 20)

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")

    ax.set_title(date.strftime("%d %B %Y"))
    ax.set_theta_zero_location("S")
    ax.set_xticklabels(["00 MLT", "", "06", "", "12", "", "18", ""])
    ax.set_rgrids([90, 80, 70, 60, 50], angle=135)
    ax.set_rlim(bottom=90, top=50)

    theta, r = np.meshgrid(mlon, mlat)
    cax = ax.contourf(
        np.deg2rad(theta),
        r,
        data,
        levels=np.linspace(0, 1, 51),
        cmap="jet",
        vmin=0,
        vmax=1,
        extend="both",
    )
    cbar = fig.colorbar(
        cax, ticks=np.linspace(0, 1, 6), orientation="horizontal", pad=0.1
    )
    cbar.set_label("ROTI, TECU/min")
