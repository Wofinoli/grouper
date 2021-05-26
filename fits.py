import json
import numpy as np

class Fit_Handler():
    def __init__(self):
        self.make_fits()
        print(self.choose_fit('INA_IV'))

    def make_fits(self):
        with open('fits.json') as f:
            self.fits = json.load(f)

    def list_fits(self):
        for fit, properties in self.fits.items():
                print(fit)

    def choose_fit(self, name):
        fit = self.fits[name]
        return Fit(name, fit['variables'], fit['control'], fit['function'])

class Fit():

    def __init__(self, _name, _variables, _control, _function):
        self.name = _name
        self.variables = _variables
        self.control = _control
    
    def __str__(self):
        return "{}\t{}\t{}".format(self.name, self.variables, self.control)
