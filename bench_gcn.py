import subprocess
import os
import time

from psutil import TimeoutExpired

dir = "../af_data/dataset_af/"
max_time = 60
t1 , t2= 0, 0
n_timeout1, n_error1, n_timeout2, n_error2 = 0, 0, 0, 0


for file in os.listdir(dir):
    #if file.startswith("adm"):
        #continue
    path = os.path.join(dir, file)
    print(file)
    try:
        tic = time.perf_counter()
        code = subprocess.call(["py", "afgcnv3_rx.py", path], timeout=max_time)
        toc = time.perf_counter()
        t1 = toc - tic
    except subprocess.TimeoutExpired:
        t1 = "timeout"
        n_timeout1 +=1
    if code != 0:
        t1 = "error"
        n_error1 +=1
    """
    try:
        tic2 = time.perf_counter()
        code1 = subprocess.call(["py", "solver_ag.py", "--filepath", path, "--task", "DC-CO", "--argument", "1"], timeout=max_time)
        toc2 = time.perf_counter()
        t2 = toc2 - tic2
    except subprocess.TimeoutExpired:
        t2 = "timeout"
        n_timeout2 +=1
    if code1 != 0:
        t2 = "error"
        n_error2 +=1
    """
    print(t1, " ", t2)