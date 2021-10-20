import os

# Just set your paths here explicitly, no need
# to use system environment as we did, this just made it easier for us
try:
    pytorch_data = os.environ['pytorch_data']
    mimic_cxr = os.environ['mimic_cxr']
except KeyError as e:
    raise KeyError("Make sure to set the data locations in lossy/data_locations.py") from e
