#from pathlib import Path

#CONFIG_FILE_PATH = Path("config/config.yaml")
#PARAMS_FILE_PATH = Path("params.yaml")

from pathlib import Path

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
PARAMS_FILE_PATH = Path(__file__).resolve().parent.parent / "params.yaml"
