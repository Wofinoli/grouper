import dill as pickle

class Fits:
    def __init__(self, filename):
        self.load_fits(filename)

    def load_fits(self, filename):
        with open(filename, 'rb') as stored_fits:
            self.fits = pickle.load(stored_fits, -1)

    def __getitem__(self, key):
        return self.fits[key]

class Fit:
    def __init__(self, name, variables, func):
        self.name = name
        self.variables = variables
        self.func = func
