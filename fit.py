import dill as pickle

class Fits:
    def __init__(self, filename):
        self.load_fits(filename)

    def load_fits(self, filename):
        pass

class Fit:
    def __init__(self, name, variables, func):
        self.name = name
        self.variables = variables
        self.func = func

def main():
    with open('fits/fits.pkl', 'rb') as input:
        fits = pickle.load(input, -1)

    for fit in fits:
        print(fit)

main()
