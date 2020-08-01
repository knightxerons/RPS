import os

pos = 2500
neg = 1250
stages = 10
w = 40
h = 40

cmd = "opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos {} -numNeg {} -numStages {} -w {} -h {}".format(pos,neg,stages,w,h)

os.system(cmd)
