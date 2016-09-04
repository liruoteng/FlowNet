import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('model_simple/solver.prototxt')
solver.net.copy_from('model_simple/flownet_official.caffemodel')

for i in xrange(3000000):
	solver.step(1)

