import os
import cPickle

# The Helper class has a few common static methods
# that are used throughout the system.
# More methods can be added here - would be useful!

class Helper:
    def __init__(self):
        pass

    @staticmethod
    def remove_file_extension(name):
        indices = [i for i, ltr in enumerate(name) if ltr == '.']
        return name[:indices[-1]]

    @staticmethod
    def remove_file_silently(filename):
        try:
            os.remove(filename)
        except OSError:
            pass

    # this pickles a couple of items and stores them on the filesystem,
    # given the filepath to store the items at.
    # The method below this one (get_stuff) does the opposite.
    # TODO- this isn't scalable! use *args and **kwargs to make these
    # two methods accept a variable number of parameters, so that they
    # can load/store more files if the caller wants it.
    @staticmethod
    def store_stuff(item1, item1filepath, item2, item2filepath):
        with open(item1filepath, "wb") as output_file:
            cPickle.dump(item1, output_file)
        with open(item2filepath, "wb") as output_file:
            cPickle.dump(item2, output_file)
    @staticmethod
    def store_stuff_submission(item, itemfilepath):
        with open(itemfilepath, "wb") as output_file:
            cPickle.dump(item, output_file)

    @staticmethod
    def get_stuff(item1filepath, item2filepath):
        with open(item1filepath, "rb") as input_file:
            item1 = cPickle.load(input_file)
        with open(item2filepath, "rb") as input_file:
            item2 = cPickle.load(input_file)
        return item1, item2
    
    @staticmethod
    def get_stuff_submission(itemfilepath):
        with open(itemfilepath, "rb") as input_file:
            item = cPickle.load(input_file)
        return item