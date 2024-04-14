import csv

bench_dir = "../benchmarks/main/"
result_approx_path = "../iccma23_approx_results.csv"
fichier = open(result_approx_path,"r")
cr = csv.reader( fichier,delimiter=";")
reduce_fichier = open("../reduce_results2023.csv", 'w')
reduce_result = csv.writer(reduce_fichier, delimiter=";")
already_checked = []
next(cr)
for row in cr:
    task = row[0]
    name = row[1]
    answer = row[3]
    correct = row[4]
    key = task+"_"+name
    if key in already_checked:
        continue
    #if name != "Small-result_b27.af":
    #    continue
    #print(row)
    argfile = open(bench_dir+name+".arg", 'r')
    arg_id = argfile.read()
    answer_b = True
    if answer == "YES":
        answer_b = True
    elif answer == "NO":
        answer_b = False
    elif answer == "NA":
        continue
    if correct == "false":
        answer_b = not answer_b
    #print(answer, " ", answer_b)

    reduce_result.writerow([task, name, arg_id, answer_b])
    already_checked.append(key)