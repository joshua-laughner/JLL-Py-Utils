import pickle


def dump_pickle(filename, obj):
    """
    Save an object to a pickle file

    :param filename: the file name to give the pickle output file. Will be overwritten if it exists
    :type filename: str

    :param obj: the Python object to pickle

    :return: None
    """
    with open(filename, 'wb') as pkfile:
        pickle.dump(obj, pkfile, protocol=pickle.HIGHEST_PROTOCOL)


def grab_pickle(filename):
    """
    Load a pickle file

    :param filename: the pickle file to load

    :return: the Python object stored in the pickle file
    """
    with open(filename, 'rb') as pkfile:
        return pickle.load(pkfile)