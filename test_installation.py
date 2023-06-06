import importlib
import sys

sys.path.append('./intrusion-detection-system/graph-based')

# import everything to make sure all the packages installed correctly
try:
    importlib.import_module("gadget-finder.gadget-finder")
    importlib.import_module("intrusion-detection-system.graph-based.gnnDriver")
    importlib.import_module("intrusion-detection-system.path-based.provdetector")
    importlib.import_module("intrusion-detection-system.path-based.sigl")
except Exception as err:
    print(f'FAIL: {err}')
else:
    print('INSTALLATION VERIFIED')
