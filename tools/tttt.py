import csv

bench_dir = "../benchmarks/main/"
result_approx_path = "../iccma23_approx_results.csv"
fichier = open(result_approx_path,"r")
cr = csv.reader( fichier,delimiter=";")
reduce_fichier = open("../reduce_results2023.csv", 'w')
reduce_result = csv.writer(reduce_fichier, delimiter=";")
all_task = ["DC-CO", "DC-ST", "DS-PR", "DS-ST"]
already_checked = []
count = {}
for ta in all_task:
    count[ta] = 0

next(cr)
for row in cr:
    task = row[0]
    name = row[1]
    answer = row[3]
    correct = row[4]
    key = task+"_"+name
    if task not in all_task:
        continue
    if "harper++" not in row[2]: 
        continue
    #if task not in all_task:
    #    continue
    if "true" in correct:
        count[task] += 1
    #reduce_result.writerow([task, name, arg_id, answer_b])
    already_checked.append(key)
for ta in all_task:
    print(ta, " ", count[ta])