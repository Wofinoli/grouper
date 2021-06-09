import sys
import json
import numpy as np

class Fit_Handler():
    def __init__(self):
        self.make_fits()

    def make_fits(self):
        with open('fits.json') as f:
            self.fits = json.load(f)

    def list_fits(self):
        names = []
        for fit, _ in self.fits.items():
                names.append(fit)

        return names

    def choose_fit(self, name):
        fit = self.fits[name]
        return Fit(fit)

    @staticmethod
    def handle_pre(pre, values):
        print(values)


    @staticmethod
    def handle_post(posts, data):
        for post in posts: 
            if post == "norm_to_max":
                data_max = max(data)
                data = [d / data_max for d in data]
        
        return data

    @staticmethod
    def handle_p0(p0, data, ctrl):
        result = []
        for item in p0:
            try:
                num = float(item)
                result.append(num)
            except:
                if item == "max":
                    result.append(max(data))
                elif item == "median":
                    result.append(np.median(data))
                elif item == "median_ctrl":
                    result.append(np.median(ctrl))

        return result

    @staticmethod
    def make_lambda(fixed, fit):
        exp_string = "lambda {}".format(fit['variables'][-1])
        for idx in range(0, len(fit['variables']) - 1):
            exp_string += ",{}".format(fit['variables'][idx])

        exp_string += ": {}".format(fit['function'])
        print(exp_string)
        fn = eval(exp_string)
        return fn

class Fit():

    def __init__(self, _fit):
        self.fit = _fit
        self.fit['lambda'] = self.eval(_fit['function'])

    def eval(self, func_string):
        exp_string = "lambda {}".format(self['variables'][-1])
        for idx in range(0, len(self['variables']) - 1):
            exp_string += ",{}".format(self['variables'][idx])

        exp_string += ": {}".format(func_string)
        fn = eval(exp_string)
        return fn
        #return self.fsp.eval(exp_string)
    
    def make_label(self):
        #label = 'fit: $v_{rev}=%5.3f, g_{max}=%5.3f, v_{0.5}=%5.3f, v_{slope}=%5.3f$' % tuple(self.popt)
        pass

    def __getitem__(self, key):
        return self.fit[key]
