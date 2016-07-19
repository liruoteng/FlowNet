# == Caffe with FlowNet ==
# Release: 1.0
# Date: 08.02.2016
# Based on caffe (GIT hash SHA 8e8d97d6 by Jeff Donahue, 23.11.2015 04:33)

This is a release of FlowNet-S and FlowNet-C.
It comes as a fork of the caffe master branch and with a trained network,
as well as examples on how to use or train it.

To get started with FlowNet, first compile caffe, by configuring a

    "Makefile.config" (example given in Makefile.config.example)

then make with 

    $ make -j 5 all tools

Go to this folder:

    ./flownet-release/models/flownet/

From this folder you can execute the scripts we prepared:
To try out FlowNetS on a sample image pair, run

    ./demo_flownet.py S data/0000000-img0.ppm data/0000000-img1.ppm

You can also provide lists of files to run it on multiple image pairs.
To train FlowNetS with the 8 sample images that come with this package, just run:

    ./train_flownet.py S

To extend it, please modify the img1_list.txt and img2_list.txt files accordingly or adapt the python script for your needs.
Please use strong image augmentation techniques to obtain satisfactory results.



## License and Citation

Please cite this paper in your publications if you use FlowNet for your research:

    @inproceedings{DFIB15,
      author       = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz\ırba\ş and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
      title        = "FlowNet: Learning Optical Flow with Convolutional Networks",
      booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
      month        = "Dec",
      year         = "2015",
      url          = "http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15"
    }

---

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
