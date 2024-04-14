import subprocess
import os
import time
import csv

dir = "../benchmarks/main/"
#all_task = ["DC-CO","DC-ID", "DC-SST", "DC-ST","DC-STG", "DS-PR", "DS-SST", "DS-ST", "DS-STG"]
all_task = ["DS-ST", "DC-CO", "DC-ST", "DS-PR", "DS-ST",]
max_time = 39
all_time = {}
all_good = {}
all_error = {}
all_timeout = {}
all_instance = {}
for ta in all_task:
    all_time[ta] = 0
    all_good[ta] = 0
    all_error[ta] = 0
    all_timeout[ta] = 0
    all_instance[ta] = 0

t , err, tot_t, n_error, n_timeout = 0,0,0,0,0
reader = open("../reduce_results2023.csv", 'r')
cr = csv.reader(reader, delimiter=';')
writer = open("afgcnv3_results2023.csv", 'w')
cw = csv.writer(writer, delimiter='w')

for row in cr:
    task = row[0]
    instance_name = row[1]
    arg_id = row[2]
    truth_answer = eval(row[3])
    if task not in all_task:
        continue
    if task != "DS-ST":
        continue
    #if instance_name != "Small-result_b27.af":
    #    continue

    path = os.path.join(dir, instance_name)
    print("-----------------------------------------")
    print("FILE : ", instance_name, " TASK : ", task)
    try:
        tic = time.perf_counter()
        #run = subprocess.run(["python", "-W ignore", "afgcnv3.py", path, task, arg_id],stdout=subprocess.PIPE, timeout=max_time)
        #run = subprocess.run(["python", "-W ignore", "afgcnv3_f11.py", path, task, arg_id],stdout=subprocess.PIPE, timeout=max_time)
        run = subprocess.run(["python", "-W ignore", "afgcnv3_norm.py", path, task, arg_id],stdout=subprocess.PIPE, timeout=max_time)
        #run = subprocess.run(["../rapproximate_hcat/target/release/rapproximate", "-p", task,"-f", path,"-a", arg_id, "-n"],stdout=subprocess.PIPE, timeout=max_time)
        toc = time.perf_counter()
        res = run.stdout.decode('utf-8').strip()
        all_instance[task] += 1
        print(res, " ", arg_id, " ", truth_answer)
        if (res == "YES" and truth_answer == True) or (res == "NO" and truth_answer == False):
            all_good[task] +=1
            print("CORRECT : ", all_good[task]," / ", all_instance[task], " acc : ",  (all_good[task] / all_instance[task])*100, "%")
        else:
            print("INCORRECT : ", all_good[task]," / ", all_instance[task], " acc : ",  (all_good[task] / all_instance[task])*100, "%")
        t = toc - tic
        all_time[task] += t
    except subprocess.TimeoutExpired:
        t = "Timeout"
        all_timeout[task] +=1
    if run.returncode != 0:
        t = "Error"
        all_error[task] +=1

    print(t, "sec")

print("---------------------------------------------------")
print("Result Finale : AFGCNV3")
for ta in all_task:
    print("TASK : ", ta)
    print("Score : ", all_good[ta])
    print("Timeout : ", all_timeout[ta])
    print("Error : ", all_error[ta])
    print("Error : ", all_time[ta])
