# Paste into a Kaggle notebook cell (Internet ON, competition data attached).
import subprocess, sys, os
REPO = "https://github.com/tanglijiao/Kaggle-game.git"
BRANCH = "main"
CMD = "python -u scripts/00_explore.py --out /kaggle/working/explore"
os.chdir("/kaggle/working")
subprocess.run(["rm", "-rf", "repo"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, "repo"], check=True)
os.chdir("repo"); sys.path.insert(0, "kaggle")
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
from run import run; run(CMD)
