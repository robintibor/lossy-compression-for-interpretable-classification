import sys
import torch
from lossy import invglow, data_locations


def load_small_glow():
    return load_glow(data_locations.small_glow_path)

def load_normal_glow():
    return load_glow(data_locations.normal_glow_path)

def load_glow(model_path, map_location=None):
    sys.modules['invglow'] = invglow
    if not torch.cuda.is_available() and map_location is None:
        map_location = torch.device('cpu')
    gen = torch.load(model_path, map_location=map_location)
    return gen