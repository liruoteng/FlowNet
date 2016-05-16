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


