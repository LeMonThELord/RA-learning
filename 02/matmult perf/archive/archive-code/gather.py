# gather all the data from the csv directory
# and plot it in a single graph

import numpy as np
from io import StringIO

# All files are in the format: [dim] [time/s]
# Result woule be like: [dim] [time/s] [time/s] ...
# Where the first column is the dimension and the other columns are the times
# The result should also have a header like: [dim] [file_name] [file_name] ...

import pathlib
current_dir = pathlib.Path().absolute()
csv_dir = current_dir / "csv"

result_file = "gathered.csv"
header = []
result_array = 0

first = True
for source_file in csv_dir.iterdir():
    with open(source_file, 'r') as source:
        data = source.read()
        data = data.split('\n\n') 
        ds = np.loadtxt(StringIO(str(data[0])))
        if first:
            print(first)
            first = not first
            header.append("dim")
            result_array = ds[:1100, 0].reshape(-1, 1)
        header.append(source_file.stem)
        result_array = np.hstack((result_array, ds[:1100, 1].reshape(-1, 1)))



import csv
with open(result_file, 'w', newline='') as result:
    result_writer = csv.writer(result)
    result_writer.writerow(header)
    result_writer.writerows([[int(row[0]), *[f"{ele:.6f}"for ele in row[1:]]] for row in result_array.tolist()])


