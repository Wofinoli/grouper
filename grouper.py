import pandas as pd
import sys
import re
import string
import os

WELLS = 384

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

    return nav.loc['A01':, columns[bool_mask]]

def main():

    csv_files = [name for name in os.listdir("./input/") if name.endswith(".csv")]
    if len(csv_files) < 1:
        sys.exit("No csv files found! Are you in the right directory?")

    for i, name in enumerate(csv_files, start=1):
        print( "({}) {}".format(i, name) )

    index = int(input("Which file do you want to process? "))
    while index < 1 or index > len(csv_files):
        index = int(input("Try again. Which file do you want to process? "))

    filename = csv_files[index-1]
    rel_data = get_relevant("./input/" + filename)
    rel_arr = rel_data.iloc[:,0].to_numpy()

    plate_cols = 24
    plate_rows = 16
    row_names = list(string.ascii_uppercase[:plate_rows])

    clean = pd.DataFrame(data = rel_arr.reshape(plate_rows, plate_cols,order='F'), index=row_names, columns=range(1,plate_cols+1))

    os.makedirs(os.path.dirname("./output/"), exist_ok=True)
    clean.to_csv("./output/clean_" + filename)

main()
