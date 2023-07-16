#! /usr/bin/python

import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pathlib

current_dir = pathlib.Path().absolute()
csv_dir = current_dir / "csv"
plot_dir = current_dir / "plots"

sources = {
    "cuda alone":["total-cuda.csv", "overhead-cuda.csv"],
    "cuda pthread":["total-cuda.csv", "total-pthread-16.csv"],
    "cuda shared":["total-cuda.csv", "total-shared.csv"],
    "cuda shared overhead":["overhead-cuda.csv", "overhead-shared.csv"],
}

def clean_outliner_data(array, threshold=0.7):    
    # remove data that is too far from previous data and too far from next data
    cleaned = []
    for i in range(len(array)):
        if i == 0 or i == len(array) - 1:
            cleaned.append(array[i])
        else:
            # get the difference between the next value and the previous one
            diff = abs(array[i+1] - array[i-1])

            # compare the difference between the current value and the previous one
            if abs(array[i] - array[i-1]) > diff * threshold:
                cleaned.append(array[i-1] + (diff / 2))
            else:
                cleaned.append(array[i])
    return cleaned

def smooth_data(array, window_size=30):
    """Smooth an array by averaging its values in a window of size window_size"""
    array = clean_outliner_data(array)
    smoothed = []
    for i in range(len(array)):
        start = max(0, i - window_size // 2)
        end = min(len(array), i + window_size // 2)
        smoothed.append(np.mean(array[start:end]))
    return smoothed

def new_dir(path: pathlib.Path):
    # create the plot directory if it doesn't exist or clean it
    if not path.exists():
        path.mkdir()
    else:
        for file in path.iterdir():
            file.unlink()

def ppt_plot(plot_title, smoothed=False, data_range=(0, 1000), figsize=(7, 7), designed_style=False):

    # new figure
    plt.figure(figsize=figsize)

    # get related files
    files = []
    for patterns in sources[plot_title]:
        files += list(csv_dir.glob(patterns))
    files = sorted(files, key=lambda x: x.stem)

    # plot each file
    for plot_index, path in enumerate(files):
        with open(path) as f:
            data = f.read()
        data = data.split('\n\n') 
        ds = np.loadtxt(StringIO(str(data[0])))

        # chop data
        ds = np.array(ds[max(0, data_range[0]):min(len(ds), data_range[1]), :])

        # smooth data
        if smoothed:
            ds[:,1] = smooth_data(ds[:,1])

        label = "-".join(path.stem.split('-')[:]).replace('pthread', 'p').replace('row-cuda', 'rc')

        custom_color = None
        thread_count = label.split('-')[0]
        
        if designed_style:
            if thread_count == "1":
                custom_color = "red"
            elif thread_count == "8":
                custom_color = "blue"
            elif thread_count == "16":
                custom_color = "green"

        custom_linestyle = None
        if designed_style:
            if "rc" in label:
                custom_linestyle = "dashed"
            elif "p" in label:
                custom_linestyle = "dotted"
            elif "cuda" in label:
                custom_linestyle = "-."

        plt.plot(ds[:,0],ds[:,1],label=label, zorder=plot_index, linestyle=custom_linestyle, color=custom_color)
    
    plt.title(plot_title)
    plt.legend(loc='upper left')
    plt.xlabel("Matrix size")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.grid(which='both', axis='both')
    plt.savefig(plot_dir/f"{plot_title}.png", format="png")

if __name__ == "__main__":
    
    new_dir(plot_dir)

    ppt_plot("cuda pthread", smoothed=True, data_range=(0, 500), figsize=(7, 7))
    ppt_plot("cuda alone", smoothed=True, data_range=(0, 10000), figsize=(10, 10))
    ppt_plot("cuda shared", smoothed=True, data_range=(0, 10000), figsize=(7, 7))
    ppt_plot("cuda shared overhead", smoothed=True, data_range=(0, 10000), figsize=(7, 7))

    # for figure_index, plot_title in enumerate(sources):
    #     ppt_plot(plot_title)