import pandas as pd
import sys
import re
import string
import os

WELLS = 384

class Group:

    def __init__(self, cords, name, mean, dev, err, low, high, median, n):
        self.cords = cords
        self.name = namme
        self.mean = mean
        self.dev = dev
        self.err = err
        self.low = low
        self.high = high
        self.median = median
        self.n = n

    def __str__(self):
        str = ("{}:\n"
               "\tName:\t{}"
               "\tMean:\t{}"
               "\tDev:\t{}"
               "\tErr:\t{}"
               "\tLow:\t{}"
               "\tHigh:\t{}"
               "\tMedian:\t{}"
               "\tN:\t{}").format(
                   self.cords, self.name,
                   self.mean, self.dev,
                   self.err, self.low,
                   self.high, self.median,
                   self.n)
        return str


def choose_file():
    csv_files = [name for name in os.listdir("./input/") if name.endswith(".csv")]
    if len(csv_files) < 1:
        sys.exit("No csv files found! Are you in the right directory?")

    for i, name in enumerate(csv_files, start=1):
        print( "({}) {}".format(i, name) )

    index = int(input("Which file do you want to process? "))
    while index < 1 or index > len(csv_files):
        index = int(input("Try again. Which file do you want to process? "))

    return csv_files[index-1]


# Gets relevant data from the csv file
# Needs: filename
# Returns: Dataframe with relevant data
def get_relevant(filename):
    print("Getting relevant data.")
    nav = pd.read_csv(filename,sep='\t', index_col = 0)

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

def main():

    # Choose file
    filename = choose_file()
    rel_data, parameters, num_sweeps = get_relevant("./input/" + filename)

    os.makedirs(os.path.dirname("./output/"), exist_ok=True)

    # Choose plate layout
    plate_options = [(24,16)]
    for i, options in enumerate(plate_options, start=1):
        print("({}) {} columns, {} rows".format(i, options[0], options[1]))

    option = int(input("What is your plate layout? " ))
    while option < 1 or option > len(plate_options):
        option = int(input("That's not an option. Try again: "))

    plate_cols, plate_rows = plate_options[option-1][0], plate_options[option-1][1]

    row_names = list(string.ascii_uppercase[:plate_rows])

    if not plate_cols * plate_rows == WELLS:
        sys.exit("The number of columns and rows must multiply to 384.")

    # Create directories for each parameter
    path = "./output/" + filename[:-4] + "/"
    for index, param in enumerate(parameters, start=0):
        param = param.replace("/", "Per")
        param_path = path + param + "/"
        if not os.path.isdir(param_path):
            os.makedirs(param_path, exist_ok=True);

        for sweep in range(1, num_sweeps+1):
            filepath = param_path + "/Sweep{:03}.csv".format(sweep)
            rel_arr = rel_data.iloc[:,len(parameters)*(sweep-1) + (index-1)].to_numpy()
            clean = pd.DataFrame(data = rel_arr.reshape(plate_rows, plate_cols,order='F'), index=row_names, columns=range(1,plate_cols+1))

            clean.to_csv(filepath)

    # Choose sweep
#    sweep = int(input("\nThere are {} sweeps. Which sweep do you want to process? ".format(num_sweeps)))
#    while sweep < 1 or sweep > num_sweeps:
#         sweep = int(input("Try again. Which sweep do you want to process? "))
#    filename = filename[:-4] + "_Sweep{:03}".format(sweep)
#
#    # Choose parameter
#    for i, param in enumerate(parameters, start=1):
#        print("({}) {}".format(i, param))
#
#    index = int(input("\nWhich parameter do you want to process? "))
#    while index < 1 or index > len(parameters):
#        index = int(input("Try again. Which parameter do you want to process? "))
#    filename += "_{}".format(parameters[index-1])
#
#    rel_arr = rel_data.iloc[:,len(parameters)*(sweep-1) + (index-1)].to_numpy()
#
#    clean = pd.DataFrame(data = rel_arr.reshape(plate_rows, plate_cols,order='F'), index=row_names, columns=range(1,plate_cols+1))
#
#    clean.to_csv("./output/clean_" + filename + ".csv")

main()
