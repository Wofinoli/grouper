class Group:

    def __init__(self, _color, _name):
        self.coordinates = []
        self.color = _color
        self.name = _name

    def to_string(self):
        return "Color: " + self.color

    def add_cords(self, cords):
        self.coordinates.append(cords)

