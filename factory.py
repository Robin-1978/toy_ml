import importlib

def load_class_from_string(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def create_model_from_config(model):
    module_name = model["model"]["module"]
    class_name = model["model"]["name"]
    model_class = load_class_from_string(module_name, class_name)
    params = model["model"]["param"]
    return model_class(**params)

def load_data_from_config(model):
    return load_class_from_string(model["module"], model["data"])()

def model_list(models):
    return [model["name"] for model in models]