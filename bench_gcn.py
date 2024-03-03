import subprocess
import os
import time

from psutil import TimeoutExpired

dir = "../af_data/dataset_af/"
max_time = 60
t1 , t2= 0, 0

for file in os.listdir(dir):
    path = os.path.join(dir, file)
    print(file)
    try:
        tic = time.perf_counter()
        code = subprocess.call(["py", "afgcnv3.py", path], timeout=max_time)
        toc = time.perf_counter()
        t1 = toc - tic
    except subprocess.TimeoutExpired:
        t1 = "timeout"
    
    try:
        tic2 = time.perf_counter()
        code1 = subprocess.call(["py", "solver_ag.py", "--filepath", path, "--task", "DC-CO", "--argument", "1"], timeout=max_time)
        toc2 = time.perf_counter()
        t2 = toc2 - tic2
    except subprocess.TimeoutExpired:
        t2 = "timeout"

    print(t1, " ", t2)