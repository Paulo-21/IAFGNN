import os
import subprocess

dir = "../../../Documents/dataset/"

for l in os.listdir(dir):
    if l.endswith(".apx"):
        return_code = subprocess.call("../ICCMAfmtconverter/target/release/ICCMAfmtconverter.exe 1 "+dir+l, shell=False)
        print(l)
        print(return_code)