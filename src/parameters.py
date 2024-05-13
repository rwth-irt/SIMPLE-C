import json
import pathlib

params = None


def init(filename=None):
    global params
    if filename is None:
        filename = pathlib.Path(__file__).parent.parent.absolute() / "default_params.json"
    with open(filename, "r") as f:
        params = json.load(f)
