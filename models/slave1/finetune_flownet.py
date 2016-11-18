#! /usr/bin/python

import sys
import caffe

if len(sys.argv) > 1:
	device_no = sys.argv[1]
else:
	device_no = 0

caffe.set_mode_gpu()
caffe.set_device(int(device_no))
net = caffe.Net('model_simple/train.prototxt', 'model_simple/flownet_official.caffemodel', caffe.TRAIN)
solver = caffe.SGDSolver('model_simple/solver.prototxt')

for i in xrange(3000000):
    solver.step(1)

