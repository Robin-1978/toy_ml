import importlib

def load_class_from_string(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def create_model_from_config(config):
    module_name = config["model"]["module"]
    class_name = config["model"]["name"]
    model_class = load_class_from_string(module_name, class_name)
    params = config["model"]["param"]
    return model_class(**params)
