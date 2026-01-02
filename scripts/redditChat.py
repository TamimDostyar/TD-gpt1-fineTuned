import csv
import pandas as pd
import pathlib
import sys
import os

parent_dir = pathlib.Path(__file__).resolve().parent.parent

data = pd.read(str(parent_dir/"data/casual_data_windows.csv"))
print(data)