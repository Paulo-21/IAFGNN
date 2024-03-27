import subprocess
import os
import time

dir = "../benchmarks/main/"
task = "DC-CO"
max_time = 39
t , err, tot_t, n_error, n_timeout = 0,0,0,0,0


for file in os.listdir(dir):
    if file.endswith(".arg"):
        continue
    path = os.path.join(dir, file)
    arg_path = path+".arg"
    arg_flux = open(arg_path ,mode='r')
    arg_id = arg_flux.read()
    arg_flux.close()
    print("FILE : ", file)
    try:
        tic = time.perf_counter()
        code = subprocess.call(["python", "afgcnv3.py", path, task, arg_id], timeout=max_time)
        toc = time.perf_counter()
        t = toc - tic
        tot_t+=t
    except subprocess.TimeoutExpired:
        t = "Timeout"
        n_timeout +=1
    if code != 0:
        t = "Error"
        n_error +=1

    print(t, "sec")

print("---------------------------------------------------")
print("Result Finale : Paul")
print("NB Error   : ", n_error)
print("NB Timeout : ", n_timeout)
print("Tot time   : ", tot_t)
