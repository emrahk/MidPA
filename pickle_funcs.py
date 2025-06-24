import pickle

def save_session(filename, var_dict):
    """
    Save variables from a dictionary to a pickle file.
    
    Parameters:
        filename (str): File to save variables into.
        var_dict (dict): Dictionary containing variables to save.
    """
    with open(filename, 'wb') as f:
        pickle.dump(var_dict, f)
    print(f"Session saved to {filename}")

def load_session(filename, global_vars):
    """
    Load variables from a pickle file into a provided namespace (like globals()).
    
    Parameters:
        filename (str): File to load variables from.
        global_vars (dict): Namespace where to load variables (typically globals()).
    """
    with open(filename, 'rb') as f:
        loaded_vars = pickle.load(f)
    global_vars.update(loaded_vars)
    print(f"Session restored from {filename}. Variables loaded: {list(loaded_vars.keys())}")
