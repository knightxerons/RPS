import os

path = os.listdir("./Negative_128X128")
os.system("rm -rf bg.txt;touch bg.txt")

for i in path:
    line = "./Negative_128X128/"+i+"\n"
    with open ("bg.txt",'a') as f:
        f.write(line)

