import json
import pathlib

class Params:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Params, cls).__new__(cls)
            cls._instance.params = {}
        return cls._instance

    def init(self, filename=None):
        if filename is None:
            filename = pathlib.Path(__file__).parent.parent.absolute() / "default_params.json"
        with open(filename, "r") as f:
            self.params = json.load(f)

    def get(self, key, default=None):
        return self.params.get(key, default)

# Global instance
params_instance = Params()

# Functions to interact with the global instance
def init(filename=None):
    params_instance.init(filename)

def get_param(key, default=None):
    return params_instance.get(key, default)
