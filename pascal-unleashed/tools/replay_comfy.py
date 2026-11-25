"""Replay the original model workflows with the release DLL."""

from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'work'/'anima_p40'))
import run_four

run_four.__file__=str(HERE/'run_four.py')
run_four.main()
