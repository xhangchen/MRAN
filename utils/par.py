
class Struct:  # dict -> class object
    def __init__(self, **entries):
        self.__dict__.update(entries)