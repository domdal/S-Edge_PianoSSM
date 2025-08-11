import os

try:
    print('Virtual enviourment at:', os.environ['VIRTUAL_ENV'])

except KeyError:
    print('No virtual environment detected. Please activate your virtual environment.')
    exit(1)


import sys
import subprocess

# Upgrade pip
subprocess.check_call(['git', 'submodule', 'init'])

# Upgrade pip
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

# Install required packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'req.txt'])

import shutil

import cosine_annealing_warmup

shutil.copy2('./patches/scheduler.py', cosine_annealing_warmup.__path__[0])