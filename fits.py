import sys
import json
import numpy as np
import FunctionStringParser

class Fit_Handler():
    def __init__(self):
        self.make_fits()

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
        self.func_string = _function
        self.fsp = FunctionStringParser.FunctionStringParser()

    def eval(self, args):
        num_var = len(self.variables)
        num_arg = len(args)
        if(num_var != num_arg):
            sys.exit("Not the right amount of arguments. Got {} but expected {}.".format(num_arg,num_var))

        exp_string = "" 
        for idx in range(0, num_var):
            if(len(exp_string) == 0):
                exp_string = self.func_string

            exp_string = exp_string.replace(self.variables[idx], str(args[idx]))

        return self.fsp.eval(exp_string)

    
    def __str__(self):
        return "{}\t{}\t{}".format(self.name, self.variables, self.control)
