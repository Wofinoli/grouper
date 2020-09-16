import pandas as pd
import re
import string
import os

WELLS = 384

# Gets relevant data from the csv file
# Needs: filename
# Returns: Dataframe with relevant data
def get_relevant():
    print("Getting relevant data.")
    nav = pd.read_csv('Nav18_virtSSI_15.53.46 RF091120.csv',sep='\t', index_col = 0)

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

    csv_files = [name for name in os.listdir("./") if name.endswith(".csv")]
    print(csv_files)

    rel_data = get_relevant()
    rel_arr = rel_data.iloc[:,0].to_numpy()

    plate_cols = 24
    plate_rows = 16
    row_names = list(string.ascii_uppercase[:plate_rows])

    clean = pd.DataFrame(data = rel_arr.reshape(plate_rows, plate_cols,order='F'), index=row_names, columns=range(1,plate_cols+1))

    os.makedirs(os.path.dirname("./out/"), exist_ok=True)
    clean.to_csv('./out/clean.csv')

main()
