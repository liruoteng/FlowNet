#! /usr/python/bin
# deep vis
import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image
import os



class Visualizer():
	'''A generic visualizer for CNN'''

	def __init__(self, model=None, weights=None, image=None, label=None):
		self.env_setting()
		self.model = model          # Your model prototxt file
		self.weights = weights      # your pre-trained .caffemodel file
		self.image = image
		self.label = label
		self.net = None


	def env_setting(self):
		caffe.set_mode_gpu()
		caffe.set_device(0)


	def static_setup(self):
		# easy setup
		self.model = 'tmp/deploy.prototxt'
		self.weights = 'model_simple/flownet_official.caffemodel'
		self.net = caffe.Net(self.model, self.weights, caffe.TEST)
		self.net.forward()
		return self.net


	def setup_net(self, model=None, weights=None):
		# process input images
		# TODO: modify dynamically generated input source file
		# img = data_processing(f1,f2)
		# net.blobs['img0'].data[...] = img[0]
		# net.blobs['img1'].data[...] = img[1]
		# Initialize network
		if model is not None and weights is not None:
			self.net = caffe.Net(model, weights, caffe.TEST)
		elif self.model is not None and self.weights is not None:
			self.net = caffe.Net(self.model, self.weights, caffe.TEST)
		else:
			raise Exception('Please provide .prototxt file and .caffemodel you wanna visualize')
		return self.net


	def flatten_neurons(self, blob, axis=2):
		cube  = blob[0]
		if axis==2:
			cube = cube.transpose((1,2,0))
		elif axis==1:
			cube = cube.transpose((1,0,2))
		elif axis==0:
			cube = cube.transpose((2,0,1))
		else:
			raise Exception("3-dimensional blob doesn't have this axis: %i" % axis)

		[height, width, channel] = cube.shape
		panel_width = np.ceil(np.sqrt(channel)).astype(np.uint8) 
		if channel % panel_width == 0:
			panel_height = int(channel/panel_width)
		else:
			panel_height = int(channel / panel_width) + 1

		panel = np.zeros((panel_height*height, panel_width*width))
		for i in range(channel):
			patch = cube[:, :, i]
			col = i % panel_width
			row = int(i / panel_width)
			panel[row*height:(row+1)*height, col*width:(col+1)*width] = patch

		return panel


	def visualize(self, blobname, mode='display', out='out', axis=2):
		blob = self.net.blobs[blobname].data
		panel = self.flatten_neurons(blob, axis=axis)
		if mode=='display':
			plt.imshow(panel)
			plt.show()
		elif mode == 'save':
			plt.figure(figsize=(16,12))
			plt.imshow(panel)
			plt.savefig(os.path.join('out', blobname + '.jpg'), bbox_inches='tight')
			plt.close('all')
		else:
			raise Exception('No such mode %s' % mode)


	def visualize_all(self, out='out', axis=2):
		if not os.path.exists(out):
			os.makedirs(out)
		for blobname in self.net.blobs:
			self.visualize(blobname, mode='save', out=out, axis=axis)

			
	def data_processing(f1,f2):
		# interface for additional preprocessing steps here
		# this functiom is currently not necessary for flownet archi
		img0 = read_image(f1)
		img1 = read_image(f2)
		return (img0, img1)


	def read_image(f):
		# read image file
		img = np.array(Image.open(f))
		# RGB to BGR for caffe
		im = img[:, :, ::-1]
		im = im.transpose((2,0,1))
		return im


if __name__ == '__main__':
    print 'AH!! You probably want to run ./run_vis.py instead.:)'