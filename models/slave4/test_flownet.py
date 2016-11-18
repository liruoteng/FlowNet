#! /usr/bin/python

import os, sys
from PIL import Image
from scripts import kittitool
from scripts import flowlib as fl
from scripts.flownet import FlowNet


this_dir = os.path.dirname(os.path.realpath(__file__))

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


def test_sintel():
	pass


def test_kitti():
	
	sum_error = 0
	sum_px_error = 0
	result = open('result.txt', 'wb')
	prediction_file = 'flownets-pred-0000000.flo'
	img1_list = open('img1_kitti_test.txt', 'r').readlines()
	img2_list = open('img2_kitti_test.txt', 'r').readlines()
	flow_list = open('flo_kitti_test.txt', 'r').readlines()
	length = len(img1_list)

	for i in range(length):
		# input images and ground truth flow
		img_files = []
		img_files.append(img1_list[i].strip())
		img_files.append(img2_list[i].strip())

		# sanity check
		if os.path.exists(prediction_file):
			os.remove(prediction_file)

		#invoke FlowNet
		FlowNet.run(this_dir, img_files, './model_simple')

		#evaluate result
		flow = fl.read_flow(prediction_file)
		gt = kittitool.flow_read(flow_list[i].strip())
		epe = fl.evaluate_flow(gt, flow)
		sum_error += epe

		# write to result file
		result.write(str(i) + ':\n' + img_files[0] +'\n' + img_files[1] + '\n' + flow_list[i].strip() + '\n')
		result.write('Average end point error: ' + str(epe) + '\n')


	result.write('Total average point error: ' + str(sum_error) + '\n')
	print 'sum of average end point error: ', sum_error
	result.close()


def test_rain():
	rain_image_path = 'haze_rain'
	prediction_file = 'flownets-pred-0000000.flo'
	left_name_base = 'haze_rain_light/render_haze_left_beta'
	right_name_base = 'haze_rain_light/render_haze_right_beta'
	flow_file = 'haze_rain_light/flow_left.flo'
	result = open('result.txt', 'wb')
	sum_error = 0
	for beta in range(0, 200, 5):
		for contrast in range(120, 201, 5):
			img_files = []
			left_name =  left_name_base + str(beta) + 'contrast' + str(contrast) + '.png'
			right_name = right_name_base + str(beta) + 'contrast' + str(contrast) + '.png'
			img_files.append(right_name)
			img_files.append(left_name)

			# sanity check
			if os.path.exists(prediction_file):
				os.remove(prediction_file)

			FlowNet.run(this_dir, img_files, './model_simple')
			epe = fl.evaluate_flow_file(flow_file, prediction_file)
			flow = fl.read_flow(prediction_file)
			flowpic = fl.flow_to_image(flow)
			flow_image = Image.fromarray(flowpic)
			flow_image.save('beta' + str(beta)+ 'contrast' + str(contrast) + 'flow.png')
			
			sum_error += epe

			result.write('beta: ' + str(beta) + ' contrast: ' + str(contrast) + ' epe: ' + str(epe) + '\n')


	print 'sum of average end point error: ', sum_error
	result.close()


def test_flownet():
	sum_error = 0
	sum_px_error = 0
	result = open('result.txt', 'wb')
	prediction_file = 'flownets-pred-0000000.flo'
	img1_list = open('img1_list_test.txt', 'r').readlines()
	img2_list = open('img2_list_test.txt', 'r').readlines()
	flow_list = open('flo_list_test.txt', 'r').readlines()
	length = len(img1_list)
	
	for i in range(length):
		img_files = []
		img_files.append(img1_list[i].strip())
		img_files.append(img2_list[i].strip())
		# sanity check
		if os.path.exists(prediction_file):
			os.remove(prediction_file)
		FlowNet.run(this_dir, img_files, './model_simple')
		epe = fl.evaluate_flow_file(flow_list[i].strip(), prediction_file)
		flow = fl.read_flow(prediction_file)
		[height, width, channels] = flow.shape
		sum_error += epe
		sum_px_error += epe / (height * width)
		result.write(str.format("%4d" % i) + ': ' + str(epe) + '\n')


	print 'Average Image EPE error: ', sum_error/length
	print 'Average Pixel EPE error: ', sum_px_error/length
	result.write('\n')
	result.write('Average Image EPE error: ' + str(sum_error / length))
	result.write('Average Pixel EPE error: ' + str(sum_px_error / length))
	result.close()


if __name__ == '__main__':
	if len(sys.argv) == 1:
		test_flownet()
	elif len(sys.argv) == 2:
		if sys.argv[1] == 'KITTI':
			test_kitti()
		elif sys.argv[1] == 'Middlebury':
			test_middlebury()
		elif sys.argv[1] == 'Sintel':
			test.sintel()
		else:
			raise error("argument not supported")
	else:
		test_flownet()



