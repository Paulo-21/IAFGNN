import af_reader_py
import sys
from DatasetDGL import transfom_to_graph
import torch 
torch.set_printoptions(profile="full")
solution_path = sys.argv[1]
file_path = sys.argv[2]

new = file_path.split("/")[-1].replace("tgf", "apx")
print(new)

all_ds = af_reader_py.read_lars_solution_ds(solution_path, file_path)
target = transfom_to_graph("../af_dataset/all_result/result_DS-ST_2017/"+new, 300)

print(all_ds)
print("TARGET : ")
print(target.numpy().tolist())