#!/usr/bin/python

import os, sys
from scripts.flownet import FlowNet

my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir + '/..')

if len(sys.argv)-1 != 3:
    print("Use this tool to test FlowNet on images\n"
          "Usage for single image pair:\n"
          "    ./demo_flownet.py {S|C} IMAGE1 IMAGE2\n"
          "\n"
          "Usage for a pair of image lists (must end with .txt):\n"
          "    ./demo_flownet.py {S|C} LIST1.TXT LIST2.TXT\n")
    sys.exit(1)

model_folder = ''
if sys.argv[1].upper() == 'S':
    model_folder = './model_simple'
elif sys.argv[1].upper() == 'C':
    model_folder = './model_corr'
else:
    print("Please specify S for FlowNetSimple or C for FlowNetCorr\n")
    sys.exit(1)

img_files = sys.argv[2:]

FlowNet.run(my_dir, img_files, model_folder)
