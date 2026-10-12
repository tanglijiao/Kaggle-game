# Paste into a Kaggle notebook cell (Internet ON, GPU optional, competition data attached).
# Edit REPO / BRANCH / CMD; everything else lives in the repository.
import subprocess, sys, os
REPO = "https://github.com/tanglijiao/Kaggle-game.git"
BRANCH = "main"
CMD = "python scripts/00_explore.py --out /kaggle/working/explore"

os.chdir("/kaggle/working")
subprocess.run(["rm", "-rf", "repo"], check=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, "repo"], check=True)
os.chdir("repo")
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)
subprocess.run(CMD, shell=True, check=True)
