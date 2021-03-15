import os, sys

PYTHONHOME = os.environ['PYTHONHOME'] if 'PYTHONHOME' in os.environ else None
if PYTHONHOME:
    PYTHON_SCRIPT_PATH = os.path.join(PYTHONHOME, 'Scripts')
    if PYTHON_SCRIPT_PATH not in sys.path:
        sys.path.append(PYTHON_SCRIPT_PATH)
else:
    print("PYTHONHOME not set, not adding PYTHONHOME/Scripts to sys.path")
