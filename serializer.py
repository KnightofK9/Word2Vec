import json
import jsonpickle

import utilities


class BaseSerialize:
    def __init__(self):
        pass

    def save(self, obj, path):
        pass

    def load(self, path):
        pass


class JsonSerialize(BaseSerialize):
    def __init__(self):
        BaseSerialize.__init__(self)

    def save(self, obj, path):
        with open(path, 'w') as outfile:
            json.dump(obj, outfile)

    def load(self, path):
        with open(path, 'r') as infile:
            return json.load(infile)


class JsonClassSerialize(BaseSerialize):
    def __init__(self):
        BaseSerialize.__init__(self)

    def save(self, obj, path):
        encode = jsonpickle.encode(obj)
        utilities.save_string(encode, path)

    def load(self, path):
        encode = utilities.load_string(path)
        return jsonpickle.decode(encode)
