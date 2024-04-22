import csv

bench_dir = "../benchmarks/main/"
result_approx_path = "../iccma23_approx_results.csv"
fichier = open(result_approx_path,"r")
cr = csv.reader( fichier,delimiter=";")
reduce_fichier = open("../reduce_results2023.csv", 'w')
reduce_result = csv.writer(reduce_fichier, delimiter=";")
all_task = ["DC-CO", "DC-ST", "DS-PR", "DS-ST"]
already_checked = []
next(cr)
for row in cr:
    task = row[0]
    name = row[1]
    answer = row[3]
    correct = row[4]
    solver = row[2]
    key = task+"_"+name
    if "harper++" not in solver:
        continue
    #if task not in all_task:
    #    continue
    if key in already_checked:
        continue
    argfile = open(bench_dir+name+".arg", 'r')
    arg_id = argfile.read()
    answer_b = True
    if answer.startswith("YES") or answer.startswith("NO"):
        if answer == "YES" :
            answer_b = True

        elif answer == "NO":
            answer_b = False
        if correct.startswith("false"):
            answer_b = not answer_b
    else:
        print("NOPPPPP : ", answer)
        continue
    
    #print(answer, " ", answer_b)

    print(task, " ", name, " ", answer, " ", correct, " -> ", answer_b)
    reduce_result.writerow([task, name, arg_id, answer_b])
    already_checked.append(key)