import pandas as pd
import numpy as np
from scipy import stats
import sys
import re
import string
import os
import warnings

WELLS = 384
PLATE_COLS = 24
PLATE_ROWS = 16
IGNOREABLE_PARAMETERS = ["Seal Resistance","Sweep VoltagePerCurrent","Sweep disregarded"]

class Plate:

    def __init__(self):
        self.cols, self.rows = Plate.choose_layout()
        self.group_names = {}
        self.num_groups = WELLS // (self.cols * self.rows)
        self.parameters = []
        self.param_frames = {}

    def is_named(self):
        if len(self.group_names) == self.num_groups:
            return True
        return False

    def add_name(self, cord, name):
        self.group_names[cord] = name

    def get_name(self, cord):
        return self.group_names[cord]

    def add_param(self, param):
        if not param in self.parameters:
            self.parameters.append(param)

    def get_parameters(self):
        return self.parameters

    def add_param_frame(self, param, frame):
        self.param_frames[param] = frame

    def get_param_frame(self, param):
        return self.param_frames[param]

    def choose_layout():
        rows = 0
        cols = 0
        while True:
            print("\nWhat is your group layout?")
            rows = int_input("How many rows per group? ")

            while rows < 1 or rows > PLATE_ROWS:
                rows = int_input("Invalid input. Try again: ")

            cols = int_input("How many cols per group? ")
            while cols < 1 or cols > PLATE_COLS:
                cols = int_input("Invalid input. Try again: ")
                
            if WELLS % (rows*cols) == 0:
                break
            else:
                print("\nThat combination is not valid, please try again.")

        return cols, rows

    def get_frames(self):
        return self.param_frames.keys()


class Group:

    def __init__(self, cords, name, mean, dev, err, low, high, median, n):
        self.cords = cords
        self.name = name
        self.mean = mean
        self.dev = dev
        self.err = err
        self.low = low
        self.high = high
        self.median = median
        self.n = n

    def __str__(self):
        str = ("{}:\n"
               "\tName:\t{}\n"
               "\tMean:\t{}\n"
               "\tDev:\t{}\n"
               "\tErr:\t{}\n"
               "\tMin:\t{}\n"
               "\tMax:\t{}\n"
               "\tMedian:\t{}\n"
               "\tN:\t{}").format(
                   self.cords, self.name,
                   self.mean, self.dev,
                   self.err, self.low,
                   self.high, self.median,
                   self.n)
        return str

def int_input(msg):
    val = input(msg)
    while True:
        try:
            val = int(val)
        except:
            val = input("Please enter a number: ")
            continue
        break
    return val

def choose_file():
    csv_files = [name for name in os.listdir("./input/") if name.endswith(".csv")]
    if len(csv_files) < 1:
        sys.exit("No csv files found! Are you in the right directory?")

    for i, name in enumerate(csv_files, start=1):
        print( "({}) {}".format(i, name) )

    index = int_input("Which file do you want to process? ")
    while index < 1 or index > len(csv_files):
        index = int_input("Try again. Which file do you want to process? ")

    return csv_files[index-1]


# Gets relevant data from the csv file
# Needs: filename
# Returns: Dataframe with relevant data
def get_relevant(filename):
    print("Getting relevant data.")
    nav = pd.read_csv(filename,sep='\t', index_col = 0)
    nav.replace([np.inf, "-Inf"], np.nan, inplace=True)

    index = nav.index
    columns = nav.columns

    num_sweeps = 1
    num_param = 0
    bool_mask = []

    temp = 0
    for column in columns:
        if re.match("Sweep \d{3}\.\d", column):
            bool_mask.append(True)
            param = int(column[-1])
            if param > temp:
                temp = param
                if num_sweeps == 1:
                    num_param += 1
            else:
                temp = 0
                num_sweeps += 1
        else:
            bool_mask.append(False)

    return nav.loc['A01':, columns[bool_mask]], nav.iloc[0,2:2+num_param], num_sweeps # Parameter selection can probably be improved

def process_clean(clean, filepath, plate):
    cols = plate.cols
    rows = plate.rows
    num_hor = PLATE_COLS // cols
    num_ver = PLATE_ROWS // rows
    groups = []

    for i in range(num_hor):
        for j in range(num_ver):
            cords = string.ascii_uppercase[i] + str(j + 1)
            # name = cords # Save name instead of asking for each parameter input("Enter group name: ")
            if not plate.is_named():
                name = input("Enter the name for group {}: ".format(cords))
                plate.add_name(cords, name)
            else:
                name = plate.get_name(cords)

            row_start = rows*j
            row_end = rows*(j+1)
            col_start = cols*i
            col_end = cols*(i+1)
            group = clean.iloc[row_start:row_end, col_start:col_end].astype('float64')
            n = np.sum(group.count())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(group)
                std = np.nanstd(group, ddof=1) # ddf=1 for Sample STD, ddf=0 for Population STD
                ste = stats.sem(group, axis=None, nan_policy="omit")

                try:
                    low = np.nanmin(group)
                except ValueError:  #raised if 'group' is empty
                    low = np.NaN

                try:
                    high = np.nanmax(group)
                except ValueError:  #raised if 'group' is empty
                    high = np.NaN

                median = np.nanmedian(group)

            groups.append( Group(
                cords, name, mean,
                std, ste, low,
                high, median, n))

    filepath = filepath[:-4] + ".txt"
    with open(filepath, "w") as text_file:
        for group in groups:
            text_file.write("{}\n".format(group))

def process_file(plate):
    # Choose file
    filename = choose_file()
    rel_data, parameters, num_sweeps = get_relevant("./input/" + filename)

    os.makedirs(os.path.dirname("./output/"), exist_ok=True)

    row_names = list(string.ascii_uppercase[:PLATE_ROWS])

    # Create directories for each parameter
    path = "./output/" + filename[:-4] + "/"
    for index, param in enumerate(parameters, start=0):
        param = param.replace("/", "Per")
        if param in IGNOREABLE_PARAMETERS:
            print("Skipping {}". format(param))
            continue

        plate.add_param(param)

        print("Processing {}".format(param))
        param_path = path + param + "/"
        if not os.path.isdir(param_path):
            os.makedirs(param_path, exist_ok=True);

        for sweep in range(1, num_sweeps+1):
            filepath = param_path + "Sweep{:03}.csv".format(sweep)
            rel_arr = rel_data.iloc[:,len(parameters)*(sweep-1) + (index-1)].to_numpy()
            clean = pd.DataFrame(data = rel_arr.reshape(PLATE_ROWS, PLATE_COLS,order='F'), index=row_names, columns=range(1,PLATE_COLS+1))
            key = "{}{:03}".format(param, sweep)
            plate.add_param_frame(key, clean)

            clean.to_csv(filepath)
            process_clean(clean, filepath, plate)


def main():
    plate = Plate()
    
    process_file(plate)

main()
