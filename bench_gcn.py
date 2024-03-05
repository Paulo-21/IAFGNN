import subprocess
import os
import time

from psutil import TimeoutExpired

dir = "../af_data/dataset_af/"
max_time = 60
t1 , t2= 0, 0
n_timeout1, n_error1, n_timeout2, n_error2 = 0, 0, 0, 0
tot_t1, tot_t2 = 0,0

for file in os.listdir(dir):
    #if file.startswith("adm"):
        #continue
    path = os.path.join(dir, file)
    print(file)
    try:
        tic = time.perf_counter()
        code = subprocess.call(["python", "afgcnv3_rx.py", path], timeout=max_time)
        toc = time.perf_counter()
        t1 = toc - tic
        tot_t1+=t1
    except subprocess.TimeoutExpired:
        t1 = "timeout"
        tot_t1+=max_time*2
        n_timeout1 +=1
    if code != 0:
        t1 = "error"
        tot_t1+=max_time*2
        n_error1 +=1
    """
    try:
        tic2 = time.perf_counter()
        code1 = subprocess.call(["python", "solver_ag.py", "--filepath", path, "--task", "DC-CO", "--argument", "1"], timeout=max_time)
        toc2 = time.perf_counter()
        t2 = toc2 - tic2
        tot_t2+=max_time*2
    except subprocess.TimeoutExpired:
        t2 = "timeout"
        n_timeout2 +=1
        tot_t2+=max_time*2
    if code1 != 0:
        t2 = "error"
        n_error2 +=1
        tot_t2+=max_time*2
    """
    print(t1, " ", t2)

print("---------------------------------------------------")
print("Result Finale : Paul vs Lars")
print("NB Error   : ", n_error1, " | ", n_error2)
print("NB Timeout : ", n_timeout1, " | ", n_timeout2)
print("Tot time   : ", tot_t1, " | ", tot_t2)