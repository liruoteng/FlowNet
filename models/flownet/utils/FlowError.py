#!/usr/bin/python
import sys
import matplotlib.pyplot as plt
from helper import *


# Check input variable
if len(sys.argv) <= 2:
    print "insufficient input arguments"
    exit(1)

# Read flow files and calculate the errors
gt = sys.argv[1]        # ground truth file
eva = sys.argv[2]        # test file
gt_flow = readflow(gt)  # ground truth flow
eva_flow = readflow(eva)     # test flow

# Calculate errors
average_pe = flowAngErr(gt_flow[:, :, 0], gt_flow[:, :, 1], eva_flow[:, :, 0], eva_flow[:, :, 1])
print "average end point error is:", average_pe

# Visualize flow
gt_img = visualize_flow(gt_flow)
eva_img = visualize_flow(eva_flow)

plt.figure(1)
plt.imshow(gt_img)
plt.figure(2)
plt.imshow(eva_img)
plt.show()

# EOF






