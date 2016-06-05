#!/usr/bin/python

import os, sys, shutil
from scripts.flownet import FlowNet

# get root directory
my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir + '/..')

# configuration
model_folder = './model_simple'
img_files = sorted(os.listdir('flownet/data/shaman_3/'))

# Running flow and store them up
for i in range(0, len(img_files)-1):
    filename_1 = os.path.join('data/shaman_3/', img_files[i])
    filename_2 = os.path.join('data/shaman_3/', img_files[i+1])
    FlowNet.run(my_dir, [filename_1, filename_2], model_folder)
    src = os.path.join(my_dir, 'flownets-pred-0000000.flo')
    dst = os.path.join(my_dir, 'out/', filename_1+'.flo')
    shutil.move(src,dst)



