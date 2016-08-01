#! /usr/python/bin
# deep vis
import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image
import os


class Visualizer():
	'''A generic visualizer for CNN'''

	def __init__(self, model=None, weights=None, image=None, label=None, outdir=None, net=None):
		self.env_setting()
		self.model = model          # Your model prototxt file
		self.weights = weights      # your pre-trained .caffemodel file
		self.image = image          # input image file
		self.label = label          # input label file
		self.outdir = None          # output directory
		self.net = None             # Caffe network
		self.norm = 1               # normalize all channels in one layer 0-false default 1(True)


	def env_setting(self):
		caffe.set_mode_gpu()
		caffe.set_device(0)


	def static_setup(self):
		# default model
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
			self.model = model
			self.weights = weights
			self.net = caffe.Net(model, weights, caffe.TEST)
		elif self.model is not None and self.weights is not None:
			self.net = caffe.Net(self.model, self.weights, caffe.TEST)
		else:
			raise Exception('Please provide .prototxt file and .caffemodel you wanna visualize')
		return self.net


	def switch_normalization(self, boolean_value):
		self.norm = boolean_value


	def save_panel(self, panel, path):
		plt.figure(figsize=(16,12))
		plt.imshow(panel, cmap='gray')
		plt.savefig(path, bbox_inches='tight')
		plt.close('all')


	def show_panel(self, panel):
		plt.imshow(panel, cmap='gray')
		plt.show()


	def flatten_neurons(self, blob, axis=2):
		if axis==2:
			pass
		elif axis==1:
			blob = blob.transpose((0,3,2,1))
		elif axis==0:
			blob = blob.transpose((0,2,1,3))
		else:
			raise Exception("3-dimensional blob doesn't have this axis: %i" % axis)

		[num, channel, height, width] = blob.shape
		patch_number = num * channel
		panel_width = 1024
		col_num = panel_width / width
		row_num = patch_number / col_num
		panel_height = row_num * height

		# scale handler
		if self.norm == 1:
			patch_min = blob.min()
			patch_max = blob.max()

		panel = np.zeros((panel_height + row_num*4, panel_width + col_num * 4))
		for n in range(num):
			for i in range(channel):
				if self.norm == 1:
					patch = self.scale_to_image(blob[n,i,:,:], patch_min, patch_max)
				else:
					original_patch = blob[n,i,:,:]
					patch_min = original_patch.min()
					patch_max = original_patch.max()
					patch = self.scale_to_image(blob[n,i,:,:], patch_min, patch_max)
				col = (n*channel + i) % col_num
				row = int((n*channel + i) / col_num)
				patch = np.lib.pad(patch, [2,2], 'constant', constant_values=(255,255))
				panel[row*(height*4):(row+1)*(height+4), col*(width*4):(col+1)*(width+4)] = patch

		return panel


	def visualize_blob(self, blobname, axis=2, mode='display'):
		blob = self.net.blobs[blobname].data
		panel = self.flatten_neurons(blob, axis=axis)
		if mode=='display':
			self.show_panel(panel)
		elif mode == 'save':
			self.save_panel(panel, os.path.join(out, blobname + '.jpg'))
		else:
			raise Exception('No such mode %s' % mode)


	def visualize_param(self, param_name, axis=2, mode='display'):
		weights = self.net.params[param_name][0].data
		bias = self.net.params[param_name][1].data
		panel = self.flatten_neurons(weights, axis=axis)
		if mode == 'display':
			self.show_panel(panel)
		elif mode == 'save':
			self.save_panel(panel, os.path.join(out, param_name + '.jpg'))
		else:
			raise Exception('No such mode %s' % mode)


	def visualize_all_param(self, axis=2):
		for param in self.net.params:
			self.visualize_param(param, axis=axis, mode='save')


	def visualize_all_blobs(self, axis=2):
		for blobname in self.net.blobs:
			self.visualize_blob(blobname, axis=axis, mode='save')


	def scale_to_image(self, patch, low_bound, high_bound):
		scaled_patch = (patch - low_bound) * 255 / (high_bound - low_bound)
		return scaled_patch.astype(np.uint8)


	# Still in progress
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