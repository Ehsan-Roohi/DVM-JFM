import runpy,sys
from pathlib import Path
sys.argv.append('--gate')
runpy.run_path(str(Path(__file__).resolve().parent/'continue.py'),run_name='__main__')
