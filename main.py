# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from datetime import datetime, timedelta, time


def load_sleep_data(path):
    data = pd.read_csv(path)
    data = data[data.Activity == "Sleep"]
    data["start_time"] = pd.to_datetime(
        data["Date and Time"], infer_datetime_format=True
    )
    data["end_time"] = pd.to_datetime(data["End Time"], infer_datetime_format=True)
    return data


def check_status(at, data):
    for _, row in data.iterrows():
        if (row["start_time"] < at) and (at < row["end_time"]):
            return "sleeping"
    return "not sleeping"


def build_dataset(
    data, start=datetime(2021, 7, 31).date(), num_days=100, step_hours=0.25
):
    data["day"] = data.start_time.apply(lambda x: (x.date() - start).days)
    g = []
    x = []
    for d in range(1, num_days + 1):
        yesterday, tomorrow = d - 1, d + 1
        df = data.query("day >= @yesterday and day <= @tomorrow")
        for h in np.arange(0, 24 + step_hours, step_hours):
            check_time = datetime.combine(start, time()) + timedelta(days=d, hours=h)
            if check_status(check_time, df) == "sleeping":
                g.append(d)
                x.append(h)
                # add weight at boundaries if sleeping
                if h == 0:
                    for h in np.arange(-1, 0, step_hours):
                        g.append(d)
                        x.append(h)
                if h == 24:
                    for h in np.arange(25, 24, -step_hours):
                        g.append(d)
                        x.append(h)
    return pd.DataFrame(dict(g=g, x=x))


def build_ridge_plot(df, title, clip_on=(0, 24)):

    pal = sns.cubehelix_palette(df.g.nunique(), rot=-0.25, light=0.8)
    text_color = tuple(pal[-1])
    g = sns.FacetGrid(df, row="g", hue="g", aspect=50, height=0.7, palette=pal)
    g.map(
        sns.kdeplot,
        "x",
        bw_adjust=0.1,
        clip_on=clip_on,
        fill=True,
        alpha=1,
        linewidth=1.5,
        gridsize=1000,
    )
    g.map(
        sns.kdeplot,
        "x",
        clip_on=clip_on,
        color="w",
        lw=2,
        bw_adjust=0.1,
        gridsize=1000,
    )

    def _label(x, color, label):
        ax = plt.gca()
        ax.text(
            -0.015,
            0.0,
            label,
            color=color,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    g.map(_label, "x")
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=clip_on)
    ax = plt.gca()
    ax.tick_params(axis="x", colors=text_color)

    @ticker.FuncFormatter
    def _time_formatter(x, pos):
        if x in (0, 24):
            return "12 am"
        if x < 12:
            return f"{x} am"
        if x == 12:
            return "12 pm"
        else:
            return f"{x - 12} pm"

    ax.xaxis.set_major_formatter(_time_formatter)
    g.figure.subplots_adjust(hspace=-0.75)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xticks=list(range(0, 25, 3)), xlabel="")
    g.set(xlim=clip_on)
    g.despine(bottom=True, left=True)
    g.fig.suptitle(title, x=0.06, y=0, color=text_color)

    return g


if __name__ == "__main__":
    path = sys.argv[1]
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font="Georgia")
    plot = (
        load_sleep_data(path)
        .pipe(build_dataset)
        .pipe(build_ridge_plot, title="the first 100 days and nights")
    )
    plot.savefig("output.png", dpi=350, bbox_inches="tight", pad_inches=1)
