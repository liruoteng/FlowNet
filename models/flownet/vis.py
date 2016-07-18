#! /usr/python/bin
# deep vis
import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image
import os


def main():
	# set up caffe environment
	env_setting()

	# images:
	f1 = ''
	f2 = ''
	# prototxt
	proto = 'tmp/deploy.prototxt'
	# model
	model = 'model_simple/flownet_official.caffemodel'
	# blob name that want to see
	blobname = 'conv6_1'

	# Run network
	net = setup_net(proto, model, f1, f2)
	out = net.forward()
	blob = net.blobs[blobname].data
	panel = flatten_neurons(blob)
	# visualize neurons
	plt.imshow(panel)
	plt.show()



def setup_net(proto, model, f1, f2):
	# process input images
	img = data_processing(f1,f2)
	# Initialize network
	net = caffe.Net(proto, model, caffe.TEST)

	net.blobs['img0'].data[...] = img[0]
	net.blobs['img1'].data[...] = img[1]

	return net

def flatten_neurons(blob):
	[num, channel, height, width] = blob.shape
	cube  = blob.transpose((2,3,1,0))
	panel_width = np.ceil(np.sqrt(channel)).astype(np.uint8) 
	if channel % panel_width == 0:
		panel_height = int(channel/panel_width)
	else:
		panel_height = int(channel / panel_width) + 1

	panel = np.zeros((panel_height*height, panel_width*width))
	for i in range(channel):
		patch = cube[:, :, i, 0]
		col = i % panel_width
		row = int(i / panel_width)
		panel[row*height:(row+1)*height, col*width:(col+1)*width] = patch

	return panel

	
def env_setting():
	caffe.set_mode_gpu()
	caffe.set_device(0)


def data_processing(f1,f2):
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


def visualize(net, blobname):
	blob = net.blobs[blobname].data
	panel = flatten_neurons(blob)
	#panel_aug = (panel - panel.min()) / (panel.max() - panel.min()) * 255
	#jpgfile = Image.fromarray(panel.astype(np.uint8))
	#jpgfile.save(os.path.join('out', blobname + '.png'))
	height = panel.shape[0]
	width = panel.shape[1]
	plt.figure(figsize=(16,12))
	plt.imshow(panel)
	plt.savefig(os.path.join('out', blobname + '.jpg'), bbox_inches='tight')
	plt.close('all')