import numpy as np
import caffe
import flowlib as fl
import os
import lmdb
from PIL import Image
VERBOSE = 0
SUBPIXELFACTOR = 4
MIN_FLOW = -4.18505
THRESHOLD = 1e8
PATCH_SIZE = 'FULL'

def data2lmdb():
	# define image and ground truth file
	train_imagefile1 = 'data/Urban3/frame10.png'  # specify 1st image file
	train_imagefile2 = 'data/Urban3/frame11.png'  # specify 2nd image file
	train_labelfile = 'gt/Urban3/flow10.flo'   # specify label file
	test_imagefile1 = 'data/Grove2/frame10.png' # specify 1st image file
	test_imagefile2 = 'data/Grove2/frame11.png'  # sepcify 2nd image file
	test_labelfile = 'gt/Grove2/flow10.flo'  # specify test label file

	# preprocessing
	train_images = preprocess_image(train_imagefile1, train_imagefile2)
	train_labels, max_label= preprocess_label(train_labelfile)
	print("Maximum number of class in training set is: ", max_label + 1)
	# Testing data
	test_images = preprocess_image(test_imagefile1, test_imagefile2)
	test_labels, test_max_label = preprocess_label(test_labelfile)
	print("Maximum number of class in testing set is: ", test_max_label + 1)

	## TRAINING
	# read image
	db = lmdb.open('train-image-lmdb-full', map_size=int(1e12))
	with db.begin(write=True) as txn:
		for i in range(len(train_images)):
			image_data = caffe.io.array_to_datum(train_images[i])
			txn.put('{:08}'.format(i), image_data.SerializeToString())
	db.close()
	
	# read label
	db = lmdb.open('train-label-lmdb-full', map_size=int(1e12))
	with db.begin(write=True) as txn:
		for i in range(len(train_labels)):
			label_data = caffe.io.array_to_datum(train_labels[i])
			txn.put('{:08}'.format(i), label_data.SerializeToString())
	db.close()

	## TESTING
	# read image
	db = lmdb.open('test-image-lmdb-full', map_size=int(1e12))
	with db.begin(write=True) as txn:
		for i in range(len(test_images)):
			image_data = caffe.io.array_to_datum(test_images[i])
			txn.put('{:08}'.format(i), image_data.SerializeToString())
	db.close()

	# read label
	db = lmdb.open('test-label-lmdb-full', map_size=int(1e12))
	with db.begin(write=True) as txn:
		for i in range(len(test_labels)):
			label_data = caffe.io.array_to_datum(test_labels[i])
			txn.put('{:08}'.format(i), label_data.SerializeToString())
	db.close()


def preprocess_data(path, mode):
	if mode == 'label':
		folders = os.listdir(path)
		folders.sort()
		for folder in folders:
			labelfile = os.path.join('gt', folder, 'flow10.flo')
			labels, max_label = preprocess_label(p)


def preprocess_image(imagefile1, imagefile2):
	# read image file
	img1 = Image.open(imagefile1)
	img2 = Image.open(imagefile2)

	# Convert image file to array
	im1 = np.array(img1)
	im2 = np.array(img2)

	# RGB to BGR for caffe
	im1 = im1[:, :, ::-1]
	im2 = im2[:, :, ::-1]

	# Concatenate
	img = np.concatenate((im1, im2), axis=2)

	# Convert to caffe blob
	img = img.transpose((2,0,1))

	# Segment image into smaller patches
	images = []
	# check if patch size is compitible to the input image
	height, width = img.shape[1], img.shape[2]
	if PATCH_SIZE == 'FULL':
		images.append(img)
	else:
		if height%PATCH_SIZE != 0 or width%PATCH_SIZE != 0:
			raise
		else:
			for i in range(0, height/PATCH_SIZE):
				for j in range(0, width/PATCH_SIZE):
					im = img[:, PATCH_SIZE*i:PATCH_SIZE*(i+1), PATCH_SIZE*j:PATCH_SIZE*(j+1)]
					images.append(im)
	
	return images


def preprocess_label(labelfile):
	# init
	max_label = -1
	labels = []
	
	# read flow file
	flow = fl.read_flow(labelfile)
	height, width = flow.shape[0], flow.shape[1]
	
	# TODO: processing vector u, horizontal flow only
	# label patch : 32 x 32
	# ground truth map size : 388 x 584 (12 x 18 patches)
	# seperate GT map into patches
	if PATCH_SIZE == 'FULL':
		label, max_label = flow2label(flow[:, :, 0], SUBPIXELFACTOR)
		labels.append(label)
	else:
		if height%PATCH_SIZE != 0 or width%PATCH_SIZE != 0:
			raise
		else:
			for i in range(0, height/PATCH_SIZE):
				for j in range(0, width/PATCH_SIZE):
					patch = flow[PATCH_SIZE*i:PATCH_SIZE*(i+1),PATCH_SIZE*j:PATCH_SIZE*(j+1),0]
					u = np.array(patch)
					# find largest displacement
					label, largest_label = flow2label(u, SUBPIXELFACTOR)
					labels.append(label)
					if largest_label > max_label:
						max_label = largest_label

	return labels, max_label


def flow2label(flow, subpixel_factor):
	# security check
	if len(flow.shape) > 2:
		raise
	# unknown flow, occlusion 
	idx = (abs(flow) > THRESHOLD)
	flow[idx] = 0
	# Convert flow to one direction
	flow_nonnegtive = flow + abs(MIN_FLOW)
	flow_nonnegtive[idx] = 0

	# discretize flow at subpixel level
	label = np.floor(flow_nonnegtive * subpixel_factor)  
	label = label.astype(np.uint8)
	
	# get the largest label
	max_label = max(-999, np.max(label))
	print("maximum label is: ", max_label)
	
	# convert to caffe format
	label = np.expand_dims(label, axis=0)

	return label, max_label


def find_max_min_flow(flow):
	# security check
	if len(flow.shape) > 2:
		raise
	# unknown flow, occlusion 
	idx = (abs(flow) > THRESHOLD)
	flow[idx] = 0

	max_flow = max(-999, np.max(flow))
	min_flow = min(999, np.min(flow))

	if VERBOSE:
		print 'max_flow: ', max_flow
		print 'min_flow: ', min_flow

	return max_flow, min_flow

def read_lmdb(database_file):
	"""
	Read lmdb data
	return content and shape
	"""
	db = lmdb.open(database_file, readonly=True)
	with db.begin() as txn:
		raw_data = txn.get(b'00000000') # get the first key value

	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_data)

	# convert string type data to actual data
	# content now is a Nx1 array
	content = np.fromstring(datum.data, dtype=np.uint8)
	shape = [datum.channels, datum.height, datum.width]

	return content, shape


