__author__ = 'shiry'


def save_dictionary(dictionary, filename):
    f = open(filename, "w")
    f.write("content = " + str(dictionary))
    f.close()