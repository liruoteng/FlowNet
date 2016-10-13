#!/usr/bin/python

import os, sys
from scripts.flownet import FlowNet

my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir + '/..')

if len(sys.argv)-1 != 1:
    print("Use this tool to train FlowNet using the train.prototxt\n"
          "Usage for FlowNetS or FlowNetC:\n"
          "    ./train_flownet.py {S|C}\n")
    sys.exit(1)

model_folder = ''
if sys.argv[1].upper() == 'S':
    model_folder = './model_simple'
elif sys.argv[1].upper() == 'C':
    model_folder = './model_corr'
else:
    print("Please specify S for FlowNetSimple or C for FlowNetCorr\n")
    sys.exit(1)

FlowNet.train(my_dir, model_folder, sys.argv[2:])