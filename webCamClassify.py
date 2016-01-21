import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import cv2

import numpy as np
import matplotlib.pyplot as plt

import caffe
from threading import Thread
from time import sleep

def main(argv):

	pycaffe_dir = os.path.dirname(__file__)

	parser = argparse.ArgumentParser()
	# Optional arguments.
	parser.add_argument(
	    "--model_def",
	    default=os.path.join(pycaffe_dir,
	            "/home/ubuntu/bvlc-caffe/models/bvlc_reference_caffenet/deploy.prototxt"),
	    help="Model definition file."
	)
	parser.add_argument(
	    "--pretrained_model",
	    default=os.path.join(pycaffe_dir,
	            "/home/ubuntu/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
	    help="Trained model weights file."
	)
	parser.add_argument(
	    "--gpu",
	default=True,
	    action='store_true',
	    help="Switch for gpu computation."
	)
	parser.add_argument(
	    "--mean_file",
	    default=os.path.join(pycaffe_dir,
	                         '/home/ubuntu/caffe/data/ilsvrc12/ilsvrc_2012_mean.npy'),
	    help="Data set image mean of [Channels x Height x Width] dimensions " +
	         "(numpy array). Set to '' for no mean subtraction."
	)
	parser.add_argument(
	    "--raw_scale",
	    type=float,
	    default=255.0,
	    help="Multiply raw input by this scale before preprocessing."
	)
	parser.add_argument(
	    "--channel_swap",
	    default='2,1,0',
	    help="Order to permute input channels. The default converts " +
	         "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

	)
	parser.add_argument(
	    "--labels_file",
	    default=os.path.join(pycaffe_dir,
	            "/home/ubuntu/caffe/data/ilsvrc12/synset_words.txt"),
	    help="Readable label definition file."
	)
	args = parser.parse_args()

	mean, channel_swap = None, None
	if args.mean_file:
	    mean = np.load(args.mean_file)
	if args.channel_swap:
	    channel_swap = [int(s) for s in args.channel_swap.split(',')]

	if args.gpu:
	    caffe.set_mode_gpu()
	    print("GPU mode")
	else:
	    caffe.set_mode_cpu()
	    print("CPU mode")

	classifier = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
	transformer = caffe.io.Transformer({'data': classifier.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data',mean.mean(1).mean(1)/255)
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	print("Reading frames from webcam...")

	time.sleep(3)

	semaphore = False

	stream = cv2.VideoCapture(0)

	with open(args.labels_file) as f:
		labels_df = pd.DataFrame([
		  {
		    'synset_id': l.strip().split(' ')[0],
		    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
               	  }
               	  for l in f.readlines()
            	])
        labels = labels_df.sort('synset_id')['name'].values
	while semaphore == False:
		(grabbed, frame) = stream.read()
		resizedImage = cv2.resize(frame, (227, 227), 0, 0).astype('float16')
		data = transformer.preprocess('data', resizedImage/255)
		classifier.blobs['data'].data[...] = data 
		start = time.time()
		out = classifier.forward()
		end = (time.time() - start)*1000
		cv2.rectangle(frame,(5,10),(450,70),(0,0,0),-1)
		cv2.putText(frame,"FF time: %dms/%dFPS" % (end,1000/end),
			(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		print("Main: Done in %.4f s." % (time.time() - start))

		scores = out['prob'][0]
    		indices = (-scores).argsort()[:5]
    		pred = labels[indices]

    		meta = [
          	  (p, '%.5f' % scores[i])
          	  for i, p in zip(indices, pred)
    		]	

    		print meta
		outstr = meta[0][0] + ', ' + meta[1][0] + ', ' + meta[2][0]

		cv2.putText(frame,outstr,
			(10,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)	
	
    		cv2.imshow('Video',frame)

    		if cv2.waitKey(1) & 0xFF == ord('q'):
    			semaphore = True

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

