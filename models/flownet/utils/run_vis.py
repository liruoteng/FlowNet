#! usr/python/bin
# RUN_VIS.PY

import vis

if __name__ == '__main__':
	v = vis.Visualize()
	v.static_setup()
	v.visualize_all()
	print ("please check ./out folder")
