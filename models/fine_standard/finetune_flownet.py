import caffe

caffe.set_mode_gpu()

caffe.set_device(1)

net = caffe.Net('model_simple/train.prototxt', 'model_simple/flownet_official.caffemodel', caffe.TRAIN)

solver = caffe.SGDSolver('model_simple/solver.prototxt')

for i in xrange(3000000):
    solver.step(1)

