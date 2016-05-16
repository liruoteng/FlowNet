#!/usr/bin/env python
import os, sys
import subprocess
from math import ceil

class FlowNet:
    caffe_bin = './bin/caffe'
    img_size_bin = './bin/get_image_size'
    template = 'deploy.tpl.prototxt'
      
    @staticmethod
    def get_image_size(filename):
        dim_list = [int(dimstr) for dimstr in str(subprocess.check_output([FlowNet.img_size_bin, filename])).split(',')]
        if not len(dim_list) == 2:
            print('Could not determine size of image %s' % filename)
            sys.exit(1)
        return dim_list


    @staticmethod
    def sizes_equal(size1, size2):
        return size1[0] == size2[0] and size1[1] == size2[1]


    @staticmethod
    def check_image_lists(lists):
        images = [[], []]

        with open(lists[0], 'r') as f:
            images[0] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
        with open(lists[1], 'r') as f:
            images[1] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

        if len(images[0]) != len(images[1]):
            print("Unequal amount of images in the given lists (%d vs. %d)" % (len(images[0]), len(images[1])))
            sys.exit(1)

        if not os.path.isfile(images[0][0]):
            print('Image %s not found' % images[0][0])
            sys.exit(1)

        base_size = FlowNet.get_image_size(images[0][0])

        for idx in range(len(images[0])):
            print("Checking image pair %d of %d" % (idx+1, len(images[0])))
            img1 = images[0][idx]
            img2 = images[1][idx]

            if not os.path.isfile(img1):
                print('Image %s not found' % img1)
                sys.exit(1)

            if not os.path.isfile(img2):
                print('Image %s not found' % img2)
                sys.exit(1)

            img1_size = FlowNet.get_image_size(img1)
            img2_size = FlowNet.get_image_size(img2)

            if not (FlowNet.sizes_equal(base_size, img1_size) and FlowNet.sizes_equal(base_size, img2_size)):
                print('The images do not all have the same size. (Images: %s or %s vs. %s)\n Please use the pair-mode.' % (img1, img2, images[0][idx]))
                sys.exit(1)

        return base_size[0], base_size[1], len(images[0])


    @staticmethod
    def train(basepath, model_folder, other_args):
        os.chdir(basepath)

        if not (os.path.isfile(FlowNet.caffe_bin) and os.path.isfile(FlowNet.img_size_bin)):
            print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
            sys.exit(1)

        args = [FlowNet.caffe_bin, 'train',
                '-model', os.path.join(model_folder, 'train.prototxt'),
                '-solver', os.path.join(model_folder, 'solver.prototxt')] + other_args
        cmd = str.join(' ', args)
        print('Executing %s' % cmd)

        subprocess.call(args)

    @staticmethod
    def run(basepath, img_files, model_folder):
        os.chdir(basepath)

        if not (os.path.isfile(FlowNet.caffe_bin) and os.path.isfile(FlowNet.img_size_bin)):
            print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
            sys.exit(1)
        
        using_lists = False
        list_length = 1

        if img_files[0][-4:].lower() == '.txt':
            print("Checking the images in your lists...")
            (width, height, list_length) = FlowNet.check_image_lists(img_files)
            using_lists = True
            print("Done.")
        else:
            print("Image files: " + str(img_files))

            # Check images

            for img_file in img_files:
                if not os.path.isfile(img_file):
                    print('Image %s not found' % img_file)
                    sys.exit(1)


            # Get image sizes and check
            img_sizes = [FlowNet.get_image_size(img_file) for img_file in img_files]

            print("Image sizes: " + str(img_sizes))

            if not FlowNet.sizes_equal(img_sizes[0], img_sizes[1]):
                print('Images do not have the same size.')
                sys.exit(1)

            width = img_sizes[0][0]
            height = img_sizes[0][1]

        # Prepare prototxt
        subprocess.call('mkdir -p tmp', shell=True)

        if not using_lists:
            with open('tmp/img1.txt', "w") as tfile:
                tfile.write("%s\n" % img_files[0])

            with open('tmp/img2.txt', "w") as tfile:
                tfile.write("%s\n" % img_files[1])
        else:
            subprocess.call(['cp', img_files[0], 'tmp/img1.txt'])
            subprocess.call(['cp', img_files[1], 'tmp/img2.txt'])

        divisor = 64.
        adapted_width = ceil(width/divisor) * divisor
        adapted_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / adapted_width
        rescale_coeff_y = height / adapted_height

        replacement_list = {
            '$ADAPTED_WIDTH': ('%d' % adapted_width),
            '$ADAPTED_HEIGHT': ('%d' % adapted_height),
            '$TARGET_WIDTH': ('%d' % width),
            '$TARGET_HEIGHT': ('%d' % height),
            '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
            '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y)
        }

        proto = ''
        with open(os.path.join(model_folder, FlowNet.template), "r") as tfile:
            proto = tfile.read()

        for r in replacement_list:
            proto = proto.replace(r, replacement_list[r])

        with open('tmp/deploy.prototxt', "w") as tfile:
            tfile.write(proto)

        # Run caffe

        args = [FlowNet.caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
                '-weights', model_folder + '/flownet_official.caffemodel',
                '-iterations', str(list_length),
                '-gpu', '0']

        cmd = str.join(' ', args)
        print('Executing %s' % cmd)

        subprocess.call(args)

        print('\nThe resulting FLOW is stored in flownets-pred-NNNNNNN.flo')
