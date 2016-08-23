#! /usr/bin/python

import os, sys
from scripts.flownet import FlowNet
from utils import flowlib as fl

this_dir = os.path.dirname(os.path.realpath(__file__))

def test_flownet():
    pass


def test_middlebury():
    # setting up
    middlebury_image = 'data/other-data'
    middlebury_flow = 'data/other-gt-flow'
    img1_name = 'frame10_rain.png'
    img2_name = 'frame11_rain.png'
    flow_name = 'flow10.flo'
    prediction_file = 'flownets-pred-0000000.flo'
    result_file = 'result.txt'
    result = open(result_file , 'wb')
    sum_error = 0

    # Retrieve image and flow information
    folder_names = os.listdir(middlebury_image)

    for folder in folder_names:
        # input images and ground truth flow
        img_files = []
        img_files.append(os.path.join(middlebury_image, folder, img1_name))
        img_files.append(os.path.join(middlebury_image, folder, img2_name))
        ground_truth_file = os.path.join(middlebury_flow, folder, flow_name)

        # sanity check
        if os.path.exists(prediction_file):
            os.remove(prediction_file)

        # invoke FlowNet
        FlowNet.run(this_dir, img_files, './model_simple')

        # evaluate result
        epe = fl.evaluate_flow_file(ground_truth_file, prediction_file)
        sum_error += epe
        print folder, " average end point error is:", epe

        # write to result file
        result.write(folder + ':\n' + img_files[0] +'\n' + img_files[1] + '\n' + ground_truth_file + '\n')
        result.write('Average end point error: ' + str(epe) + '\n')

    result.write('Total average point error: ' + str(sum_error) + '\n')
    print 'sum of average end point error: ', sum_error
    result.close()


def test_kitti():
	
	kitti_image = '../../../../../../../media/data/LRT_Flow/KITTI/data_scene_flow/training/image_2'
	kitti_flow = 'noc'
	image_list = os.listdir(kitti_image)
	flow_list = os.listdir(kitti_flow)

	prediction_file = 'flownets-pred-0000000.flo'
	result = open('result.txt', 'wb')
	sum_error = 0

	# retrieve image and flow information
	image_list.sort()
	flow_list.sort()

	for i in range(0, 308, 2):
		# input images and ground truth flow
		img_files = []
		img_files.append(os.path.join(kitti_image, image_list[i]))
		img_files.append(os.path.join(kitti_image, image_list[i+1]))
		ground_truth_file = flow_list[i/2]

		# sanity check
		if os.path.exists(prediction_file):
			os.remove(prediction_file)

		#invoke FlowNet
		FlowNet.run(this_dir, img_files, './model_simple')
		print i 
		print img_files[0] 
		print img_files[1] 
		print ground_truth_file

		#evaluate result
		epe = fl.evaluate_flow_file('noc/' + ground_truth_file, prediction_file)
		sum_error += epe

		# write to result file
		result.write(str(i) + ':\n' + img_files[0] +'\n' + img_files[1] + '\n' + ground_truth_file + '\n')
		result.write('Average end point error: ' + str(epe) + '\n')


	result.write('Total average point error: ' + str(sum_error) + '\n')
	print 'sum of average end point error: ', sum_error
	result.close()


if __name__ == '__main__':
#	test_middlebury()
	test_kitti()
