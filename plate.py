import pandas as pd

class PaginatedDataFrame:

    def __init__(self):
        self.pages = []

    def get_page(self, page):
        return self.pages[page]

    def append_page(self, page):
        self.pages.append(page)

class Plate(PaginatedDataFrame):

    def __init__(self, name):
        self.name = name
