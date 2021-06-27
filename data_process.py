import pandas as pd
import re
import string
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Button, HBox
import asyncio
import warnings
import sys

from scipy.optimize import curve_fit
#from sklearn.preprocessing import MinMaxScaler
from scipy import stats

warnings.filterwarnings("error")

class Plate():

    def __init__(self, cols, rows, filename, fit):
        self.cols = cols
        self.rows = rows
        self.filename = filename
        self.fit = fit
        self.create_frames(fit)
        self.read_file()

    def read_file(self):
        raw = pd.read_csv(self.filename, sep='\t', index_col=0)
        index = raw.index
        columns = raw.columns
        self.row_names = list(string.ascii_uppercase[:self.rows])

        self.num_sweeps = 1
        self.num_param = 0

        bool_mask = []

        temp = 0
        for column in columns:
            if re.match(r"Sweep \d{3}\.\d", column):
                bool_mask.append(True)
                param = int(column.split(".")[-1])
                if param > temp:
                    temp = param
                    if self.num_sweeps == 1:
                        self.num_param += 1
                else:
                    temp = 0
                    self.num_sweeps += 1
            else:
                bool_mask.append(False)

        parameters = raw.iloc[0,2:2+self.num_param].tolist()
        relevant_data = raw.loc['A01':, columns[bool_mask]]

        ########### SO FAR ONLY INA ############
        self.control = []
        self.var_sweeps = []

        for index, param in enumerate(parameters):
            param = param.replace("/", "Per")
            
            if param == self.fit['control']:
                for sweep in range(1, self.num_sweeps+1):
                    col = self.num_param*(sweep-1) + index
                    self.control.append(float(relevant_data.iloc[2,col]) * 1000)
                    print(self.control[-1], "from ", 2, col)
                    
                    
            if param == self.fit['dependent']:
                for sweep in range(1, self.num_sweeps+1):
                    col = self.num_param*(sweep-1) + (index)
                    var_arr = relevant_data.iloc[:,col].astype(float).to_numpy()
                    var_arr = var_arr.reshape(
                        self.rows, self.cols, order='F')
                    clean_sweep = pd.DataFrame(data = var_arr, index=self.row_names, columns=range(1,self.cols+1))
                    self.var_sweeps.append(clean_sweep * 10**12)

        self.source = pd.DataFrame(self.control, columns=["Potential"])

    def create_frames(self, fit):
        columns = ["Cell"] + fit['variables'][:-1]
        self.accepted_fits = pd.DataFrame(columns=columns)
        self.rejected_fits = pd.DataFrame(columns=columns)
        self.failed = pd.DataFrame()
        self.statistics = pd.DataFrame(index=np.arange(0,self.accepted_fits.shape[1]-1), columns=["Variable","Mean","Median","Std. Dev","Std. Err","Max","Min","N"])
     
def func_IV_NA(v, vrev, gmax, vhalf, vslope): # IV
    return (v - vrev) * gmax/(1 + np.exp((vhalf - v)/vslope))

def get_statistics(frame):
        statistics = pd.DataFrame(index=np.arange(0,frame.shape[1]-1), columns=["Variable","Mean","Median","Std. Dev","Std. Err","Max","Min","N"])
        index = 0
        for label, content in frame.items():
            if(label == "Cell"):
                continue
            
            statistics.iloc[index]["Variable"] = label
            statistics.iloc[index]["Mean"] = np.mean(content)
            statistics.iloc[index]["Median"] = np.median(content)
            statistics.iloc[index]["Std. Dev"] = np.std(content, ddof=1)
            if len(content) > 1:
                statistics.iloc[index]["Std. Err"] = stats.sem(content, axis=None, nan_policy="omit")
            statistics.iloc[index]["Max"] = np.max(content)
            statistics.iloc[index]["Min"] = np.min(content)
            statistics.iloc[index]["N"] = np.sum(content.count())
            index += 1
            
        return statistics


