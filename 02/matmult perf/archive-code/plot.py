import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

#Files are searched inside the directory "csv" that must be inside the current directory
#In every file there is the data for a single curve
#Every read filename must start with the name in "types"
#Every line in every file follows the format: [X_coord] [Y_coord]

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
    return array
    """Smooth an array by averaging its values in a window of size window_size"""
    array = clean_outliner_data(array)
    smoothed = []
    for i in range(len(array)):
        start = max(0, i - window_size // 2)
        end = min(len(array), i + window_size // 2)
        smoothed.append(np.mean(array[start:end]))
    return smoothed

plots = {
    "Single core time computation": ["total-naive.csv", "total-improved.csv"],
    "Small pthread time": ["total-pthread-1.csv", "total-pthread-8.csv"],
    "All pthread time": ["total-pthread-*.csv"],
    "Cuda-row time": ["total-row-cuda-*.csv"],
}

rows_count = (len(plots) // 4) + 1
columns_count = len(plots)//rows_count

import pathlib
current_dir = pathlib.Path().absolute()
csv_dir = current_dir / "csv"
plot_dir = current_dir / "plot"

# create the plot directory if it doesn't exist or clean it
if not plot_dir.exists():
    plot_dir.mkdir()
else:
    for file in plot_dir.iterdir():
        file.unlink()

for figure_index, plot_title in enumerate(plots):
    plt.figure(figure_index, figsize=(30, 30))
    files = []
    for patterns in plots[plot_title]:
        files += list(csv_dir.glob(patterns))
    files = sorted(files, key=lambda x: x.stem)
    for plot_index, path in enumerate(files):
        with open(path) as f:
            data = f.read()
        data = data.split('\n\n') 
        ds = np.loadtxt(StringIO(str(data[0])))
        label = "-".join(path.stem.split('-')[1:]).replace('pthread', 'p')
        if "event" in label:
            plt.plot(smooth_data(ds[:1100,0]),smooth_data(ds[:1100,1]),label=label, zorder=9999, linewidth=1, color="black", linestyle='dotted')
            continue
        plt.plot(smooth_data(ds[:1100,0]),smooth_data(ds[:1100,1]),label=label, zorder=plot_index)
    plt.title(plot_title)
    plt.legend(loc='upper left')
    plt.xlabel("Matrix size")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.grid(which='both', axis='both')
    plt.savefig(plot_dir/f"{plot_title}.png", format="png")
