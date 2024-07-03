_params: None | dict = None  # uninitialized

parameter_definitions = {
    "relative intensity threshold": "float",
    "DBSCAN epsilon": "float",
    "DBSCAN min samples": "int",
    "maximum neighbor distance": "float",
    "minimum velocity": "float",
    "window size": "int",
    "max. vector angle [deg]": "float",
    "sample_rate_Hz": "float",
    "outlier_mean_factor": "float",
    "max_point_number_change_ratio": "float",
    "normal_cosine_weight_share": "float",
    "point_number_weight_share": "float",
}


def ros_declare_parameters(rosnode):
    """
    Declares parameters in a given ROS node. Sets all default values to the ones given in default_params.

    :param rosnode: the node to declare the parameters at
    """
    try:
        from rcl_interfaces.msg import ParameterDescriptor, ParameterType
    except ImportError as ie:
        raise Exception("Not running in ROS.") from ie

    for name, typeinfo in parameter_definitions.items():
        if typeinfo == "float":
            descr = ParameterDescriptor(name=name, type=ParameterType.PARAMETER_DOUBLE)
        elif typeinfo == "int":
            descr = ParameterDescriptor(name=name, type=ParameterType.PARAMETER_INTEGER)
        elif typeinfo == "string":
            descr = ParameterDescriptor(name=name, type=ParameterType.PARAMETER_STRING)
        else:
            raise Exception(f"Parameter type '{typeinfo}' not yet implemented!")
        rosnode.declare_parameter(name=name, descriptor=descr)


def init_from_yaml(filename):
    global _params
    import yaml
    with open(filename) as f:
        try:
            loaded = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception("Could not read yaml.") from exc
    try:
        key1 = list(loaded.keys())[0]
        key2 = list(loaded[key1].keys())[0]
        if key2 != "ros__parameters":
            raise Exception("The provided YAML file does not have the right format.")
        _params = loaded[key1][key2]
    except KeyError:
        raise Exception("The provided YAML file does not have the right format.")


def init_from_rosnode(rosnode):
    global _params
    if _params is not None:
        return  # already loaded
    _params = {}
    for name, typeinfo in parameter_definitions.items():
        try:
            if typeinfo == "float":
                _params[name] = float(rosnode.get_parameter(name).get_parameter_value().double_value)
            elif typeinfo == "int":
                _params[name] = int(rosnode.get_parameter(name).get_parameter_value().integer_value)
            elif typeinfo == "string":
                _params[name] = str(rosnode.get_parameter(name).get_parameter_value().string_value)
            else:
                raise Exception(f"Parameter type '{typeinfo}' not yet implemented!")
            print(f"Loaded parameter '{name}': {_params[name]}")
        except Exception as e:
            raise Exception(f"Could not obtain parameter '{name}' from ROS") from e


def get_param(key):
    if _params is None:
        raise Exception("Parameters not loaded, call init first!")
    if key not in _params:
        raise Exception(f"Parameter '{key}' not found!")

    return _params[key]