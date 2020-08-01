import os

if os.path.exists('positives.vec'):
    os.system('rm -rf positives.vec')

number_of_vectors = 4500
width = 40 #This number should be same in train_cascade.py
height = 40 #This number should be same in train_cascade.py

cmd = 'opencv_createsamples -info info/info.lst -num {} -w {} -h {} -vec positives.vec'.format(number_of_vectors, width, height)
os.system(cmd)
