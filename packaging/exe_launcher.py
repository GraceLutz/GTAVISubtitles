"""PyInstaller belépési pont: a csomag relatív importjai miatt nem a __main__.py-t adjuk át közvetlenül."""

import sys

from gta_felirat.__main__ import run_exe

if __name__ == "__main__":
    sys.exit(run_exe())
