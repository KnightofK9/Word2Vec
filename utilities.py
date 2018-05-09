import pickle
import datetime


def save_simple_object(obj, path):
    with open(path, 'wb') as stream:  # Overwrites any existing file.
        pickle.dump(obj, stream, pickle.HIGHEST_PROTOCOL)


def load_simple_object(path):
    with open(path, 'rb') as stream:
        data = pickle.load(stream)
        return data


def print_current_datetime():
    print(datetime.datetime.now())
