#! /usr/bin/python

import os, sys
from scripts.flownet import FlowNet

model_folder = './model_simple'
my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir + '/..')
outfile = 'flownets-pred-0000000.flo'


img1 = 'Selected/BigRain1_img1.jpg'
img2 = 'Selected/BigRain1_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/HeavyRain1_img1.jpg'
img2 = 'Selected/HeavyRain1_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/HeavyRain2_img1.jpg'
img2 = 'Selected/HeavyRain2_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/LightRain1_img1.jpg'
img2 = 'Selected/LightRain1_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/LightRain2_img1.jpg'
img2 = 'Selected/LightRain2_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/SeriousRain1_img1.jpg'
img2 = 'Selected/SeriousRain1_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/SeriousRain2_img1.png'
img2 = 'Selected/SeriousRain2_img2.png'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')


img1 = 'Selected/Window1_img1.jpg'
img2 = 'Selected/Window1_img2.jpg'
img_files = [img1, img2]
FlowNet.run(my_dir, img_files, model_folder)

os.rename(outfile, img1 + '.flo')