import os

pos = './pos'
pos_path = os.listdir(pos)


a=1

if os.path.exists("./info"):
        os.system("rm -rf ./info")
os.mkdir("./info")

os.system("touch ./info/info.lst")

print("Creating the samples")


for i in pos_path:

    print(len(pos_path)-a+1)

    if not os.path.exists("./temp"):
       os.makedirs("./temp")
    b=pos+'/'+i

    cmd = "opencv_createsamples -img {} -bg bg.txt -info temp/tinfo.lst  -pngoutput info -maxxangle 0.6 -maxyangle 0.6 -maxzangle0.6 -num 2000 > /dev/null 2>&1".format(b)
    os.system(cmd)

    temp_path = os.listdir("./temp")

    for j in temp_path:
        if not j.endswith(".lst"):
            cmd = "mv ./temp/{} ./info/{}_{}".format(j,a,j)
            os.system(cmd)
        else:
            cmd = "sed -i.bak 's/^/{}_/' ./temp/{}".format(a,j)
            os.system(cmd)
            cmd = "cat ./temp/{} >> ./info/info.lst".format(j)
            os.system(cmd)
    os.system("rm -rf ./temp")

    a+=1

print("Samples created!!!")

