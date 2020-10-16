import logging

import random
import string
from PIL import Image

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.summary_ops import *


logger = logging.getLogger()
np.random.seed(2018)


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
			 padding='SAME', activation=tf.nn.relu, training=True):
	""" Define conv for generator.

	Args:
		x: Input.
		cnum: Channel number.
		ksize: Kernel size.
		Stride: Convolution stride.
		Rate: Rate for dilated conv.
		name: Name of layers.
		padding: Default to SYMMETRIC.
		activation: Activation function after convolution.
		training: If current graph is for training or inference, used for bn.

	Returns:
		tf.Tensor: output


	"""
	assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
	if padding == 'SYMMETRIC' or padding == 'REFELECT':
		p = int(rate*(ksize-1)/2)
		x = tf.pad(x, [[0,0], [p,p], [p,p], [0,0]], mode=padding)
		padding = 'VALID'
	x = tf.layers.conv2d(x, cnum, ksize, stride, dilation_rate=rate, activation=activation, padding=padding, name=name, reuse=tf.AUTO_REUSE)
	return x


@add_arg_scope
def gen_deconv(x, cnum, name='unsample', padding='SAME', training=True):
	""" Define deconv for generator.

	The deconv is defined to be a x2 resize_nearest_neighbor operation with
	additional gen_conv operation.

	Args:
		x: Input.
		cnum: Channel number.
		name: Name of layers
		training: If current graph is for training or inference, used for bn.

	Returns:
		tf.Tensor: output

	"""
	with tf.variable_scope(name):
		x = resize(x, func=tf.image.resize_nearest_neighbor)
		x = gen_conv(x, cnum, 3, 1, name=name+'_conv', padding=padding,
			training=training)
	return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):
	""" Define conv for discriminator.
	Activation is set to leaky_relu.

	Args:
		x: Input.
		cnum: Channel number.
		ksize: Kernel size.
		Stride: Convolution stride.
		name: Name of layers.
		training: If current graph is for training or inference, used for bn.

	Returns:
		tf.Tensor: output

	"""
	x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
	x = tf.nn.leaky_relu(x)
	return x


def random_bbox(config):
	""" Generate a random tlhw with configuration.

	Args:
		config: Config should have configuration including IMG_SHAPES,
			VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

	Returns:
		tuple: (top, left, height, width)

	"""
	img_shape = config.IMG_SHAPES
	img_height = img_shape[0]
	img_width = img_shape[1]
	maxt = img_height - config.VERTICAL_MARGIN - config.HEIGHT
	maxl = img_width - config.HORIZONTAL_MARGIN - config.WIDTH
	t = tf.random_uniform(
		[], minval=config.VERTICAL_MARGIN, maxval=maxt, dtype=tf.int32)
	l = tf.random_uniform(
		[], minval=config.HORIZONTAL_MARGIN, maxval=maxl, dtype=tf.int32)
	h = tf.constant(config.HEIGHT)
	w = tf.constant(config.WIDTH)
	return (t, l, h, w)


def detection_bbox_to_mask(bbox, name='dmask'):
	""" Generate mask tensor from object detection bbox.

	Args:
		bbox: configuration tuple from object detection

	Returns:
		tf.Tensor: output with shape [1, H, W, 1]
	"""
	def bbox_to_mask(img, left, top, right, bottom):
		image = cv2.imread(img, 0)

		# create a mask
		mask = np.zeros(img.shape[:2], np.uint8)
		mask[100:300, 100:400] = 255
		masked_img = cv2.bitwise_and(img,img,mask=mask)
		return masked_img

	def npmask(bbox, height, width, delta_h, delta_w):
		mask = np.zeros((1, height, width, 1), np.float32)
		h = np.random.randint(delta_h//2+1)
		w = np.random.randint(delta_h//2+1)
		mask[:, bbox[0]+h:bbox[0]+bbox[2]-h, bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
		return mask

	with tf.variable_scope(name), tf.device('/cpu:0'):
		img_shape = config.IMG_SHAPES
		height = img_shape[0]
		width = img_shape[1]
		mask = tf.py_func(
			npmask, 
			[bbox, height, width,
			config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
			tf.float32, stateful=False)
		mask.set_shape([1] + [height, width] + [1])
	return mask


def bbox2mask(bbox, config, name='mask'):
	""" Generate mask tensor from bbox.

	Args:
		bbox: configuration tuple, (top, left, height, width)
		config: Config should have configuration including IMG_SHAPEs,
			MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

	Returns:
		tf.Tensor: output with shape [1, H, W, 1]

	"""
	def npmask(bbox, height, width, delta_h, delta_w):
		mask = np.zeros((1, height, width, 1), np.float32)
		h = np.random.randint(delta_h//2+1)
		w = np.random.randint(delta_h//2+1)
		mask[:, bbox[0]+h:bbox[0]+bbox[2]-h, bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
		return mask

	with tf.variable_scope(name), tf.device('/cpu:0'):
		img_shape = config.IMG_SHAPES
		height = img_shape[0]
		width = img_shape[1]
		mask = tf.py_func(
			npmask, 
			[bbox, height, width,
			config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
			tf.float32, stateful=False)
		mask.set_shape([1] + [height, width] + [1])
	return mask


# def dbbox2mask(bbox, config, name='detection'):
# 	""" Generate mask tensor from detected bbox.

# 	Args:
# 		bbox: detected bounding box tuple, (top, left, height, width)
# 		config: Config should have configuration including IMG_SHAPES,
# 				MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

# 	Returns:
# 		tf.Tensor: output with shape [1, H, W, 1]

# 	"""
# 	def npmask(bbox, height, width, delta_h, delta_w):
# 		mask = np.zeros((1, height, width, 1), np.float32)
# 		# h = np.random.randint(delta_h//2+1)
# 		# w = np.random.randint(delta_w//2+1)
# 		mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
# 			 bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
# 		return mask

# 	with tf.variable_scope(name), tf.device('/cpu:0'):
# 		img_shape = config.IMG_SHAPES
# 		height = img_shape[0]
# 		width = img_shape[1]
# 		mask = tf.py_func(
# 			npmask,
# 			[bbox, height, width,
# 			 ],
# 			tf.float, stateful=False)
# 		mask.set_shape([1] + [height, width] + [1])
# 	return mask



def local_patch(x, bbox):
	""" Crop local patch according to bbox.

	Args:
		x: input
		bbox: (top, left, height, width)

	Returns:
		tf.Tensor: local patch

	"""
	x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
	return x


def resize_mask_like(mask, x):
	""" Resize mask like shape of x.

	Args:
		mask: Original mask.
		x: To shape of x.

	Returns:
		tf.Tensor: resized mask

	"""
	mask_resize = resize(
		mask, to_shape=x.get_shape().as_list()[1:3],
		func=tf.image.resize_nearest_neighbor)
	return mask_resize


def spatial_discounting_mask(config):
	""" Generate spatial discounting mask constant.

	Spatial discounting mask is first introduced in publication:

		Generative Image Inpainting with Contextual Attention, Yu et al.

	Args:
		config: Config should have configuration including HEIGHT, WIDTH.
			DISCOUNTED_MASK.

	Returns:
		tf.Tensor: spatial discounting mask

	"""
	gamma = config.SPATIAL_DISCOUNTING_GAMMA
	shape = [1, config.HEIGHT, config.WIDTH, 1]
	if config.DISCOUNTED_MASK:
		logger.info('Use spatial discounting l1 loss.')
		mask_values = np.ones((config.HEIGHT, config.WIDTH))
		for i in range(config.HEIGHT):
			for j in range(config.WIDTH):
				mask_values[i, j] = max(
					gamma**min(i, config.HEIGHT-i),
					gamma**min(j, config.WIDTH-j))
		mask_values = np.expand_dims(mask_values, 0)
		mask_values = np.expand_dims(mask_values, 3)
		mask_values = mask_values
	else:
		mask_values = np.ones(shape)
	return tf.constant(mask_values, dtype=tf.float32, shape=shape)


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
						fuse_k=3, softmax_scale=10., training=True, fuse=True):
	""" Contextual attention layer implementation.

	Contextual attention is first introduced in publication:
		Generative Image Inpainting with Contextual Attention, Yu et al.

	Args:
		f: Input feature to match (foreground).
		b: Input feature to match (background).
		mask: Input mask for background, indicating patches not available.
		ksize: Kernel size for contextual attention.
		stride: Stride for extracting patches from background.
		rate: Dilation for matching.
		softmax_scale: Scaled softmax for attention.
		training: Indicating if current graph is training or inference.

	Returns:
		tf.Tensor: output

	"""
	# get shapes
	raw_fs = tf.shape(f)
	raw_int_fs = f.get_shape().as_list()
	raw_int_bs = b.get_shape().as_list()
	# extract patches from background with stride and rate
	kernel = 2*rate
	raw_w = tf.extract_image_patches(
		b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
	raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
	raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])	# transpose to b*k*k*c*hw

	# downscaling foreground option: downscaling both foreground and
	# background for matching and use original background for reconstruction.
	f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
	b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)],
			func=tf.image.resize_nearest_neighbor)	# https://github.com/tensorflow/tensorflow/issues/11651
	if mask is not None:
		mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
	fs = tf.shape(f)
	int_fs = f.get_shape().as_list()
	f_groups = tf.split(f, int_fs[0], axis=0)
	# from t(H*W*C) to w(b*k*k*c*h*w)
	bs = tf.shape(b)
	int_bs = b.get_shape().as_list()
	w = tf.extract_image_patches(
		b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
	w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
	w = tf.transpose(w, [0, 2, 3, 4, 1])	# transpose to b*k*k*c*h*w
	# process mask
	if mask is None:
		mask = tf.zeros([1, bs[1], bs[2], 1])
	m = tf.extract_image_patches(
		mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
	m = tf.reshape(m, [1, -1, ksize, ksize, 1])
	m = tf.transpose(m, [0, 2, 3, 4, 1])	# transpose to b*k*k*c*hw
	m = m[0]
	mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
	w_groups = tf.split(w, int_bs[0], axis=0)
	raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
	y = []
	offsets = []
	k = fuse_k
	scale = softmax_scale
	fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
	for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
		# conv for compare
		wi = wi[0]
		wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
		yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

		# conv implementation for fuse scores to encourage large patches
		if fuse:
			yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
			yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
			yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
			yi = tf.transpose(yi, [0, 2, 1, 4, 3])
			yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
			yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
			yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
			yi = tf.transpose(yi, [0, 2, 1, 4, 3])
		yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

		# softmax to match
		yi *= mm    # mask
		yi = tf.nn.softmax(yi*scale, 3)
		yi *= mm    # mask

		offset = tf.argmax(yi, axis=3, output_type=tf.int32)
		offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
		# deconv for patching pasting
		# 3.1 paste center
		wi_center = raw_wi[0]
		yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
		y.append(yi)
		offsets.append(offset)
	y = tf.concat(y, axis=0)
	y.set_shape(raw_int_fs)
	offsets = tf.concat(offsets, axis=0)
	offsets.set_shape(int_bs[:3] + [2])
	# case1: visualize optical flow: minus current position
	h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
	w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
	offsets = offsets - tf.concat([h_add, w_add], axis=3)
	# to flow image
	flow = flow_to_image_tf(offsets)
	# case2: visualize which pixels are attended
	# flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
	if rate != 1:
		flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
	return y, flow


def test_contextual_attention(args):
	""" Test contextual attention layer with 3-channel image input
		(instead of n-channel feature).

	"""
	import cv2
	import os
	# run on cpu
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	rate = 2
	stride = 1
	grid = rate*stride

	b = cv2.imread(args.imageA)
	b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
	h, w, _ = b.shape
	b = b[:h//grid*grid, :w//grid*grid, :]
	b = np.expand_dims(b, 0)
	logger.info('Size of imageA: {}'.format(b.shape))

	f = cv2.imread(args.imageB)
	h, w, _ = f.shape
	f = f[:h//grid*grid, :w//grid*grid, :]
	f = np.expand_dims(f, 0)
	logger.info('Size of imageB: {}'.format(f.shape))

	with tf.Session() as sess:
		bt = tf.constant(b, dtype=tf.float32)
		ft = tf.constant(f, dtype=tf.float32)

		yt, flow = contextual_attention(
			ft, bt, stride=stride, rate=rate,
			training=False, fuse=False)
		y = sess.run(yt)
		cv2.imwrite(args.imageOut, y[0])


def make_color_wheel():
	RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
	ncols = RY + YG + GC + CB + BM + MR
	colorwheel = np.zeros([ncols, 3])
	col = 0
	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
	col + RY
	# YG
	colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
	colorwheel[col:col+YG, 1] = 255
	col += YG
	# GC
	colorwheel[col:col+GC, 1] = 255
	colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
	col += GC
	# CB
	colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
	colorwheel[col:col+CB, 2] = 255
	col += CB
	# BM
	colorwheel[col:col+BM, 2] = 255
	colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
	col += + BM
	# MR
	colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col+MR, 0] = 255
	return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u, v):
	h, w = u.shape
	img = np.zeros([h, w, 3])
	nanIdx = np.isnan(u) | np.isnan(v)
	u[nanIdx] = 0
	v[nanIdx] = 0
	# colorwheel = COLORWHEEL
	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)
	rad = np.sqrt(u**2+v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) / 2 * (ncols - 1) + 1
	k0 = np.floor(fk).astype(int)
	k1 = k0 + 1
	k1[k1 == ncols+1] = 1
	f = fk - k0
	for i in range(np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0-1] / 255
		col1 = tmp[k1-1] / 255
		col = (1-f) * col0 + f * col1
		idx = rad <= 1
		col[idx] = 1-rad[idx]*(1-col[idx])
		notidx = np.logical_not(idx)
		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
	return img


def flow_to_image(flow):
	""" Transfer flow map to image.
	Part of code forked from flownet.
	"""
	out = []
	maxu = -999.
	maxv = -999.
	minu = 999.
	minv = 999.
	maxrad = -1
	for i in range(flow.shape[0]):
		u = flow[i, :, :, 0]
		v = flow[i, :, :, 1]
		idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
		u[idxunknow] = 0
		v[idxunknow] = 0
		maxu = max(maxu, np.max(u))
		minu = min(minu, np.min(u))
		maxv = max(maxv, np.max(v))
		minv = min(minv, np.min(v))
		rad = np.sqrt(u ** 2 + v ** 2)
		maxrad = max(maxrad, np.max(rad))
		u = u/(maxrad + np.finfo(float).eps)
		v = v/(maxrad + np.finfo(float).eps)
		img = compute_color(u, v)
		out.append(img)
	return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
	""" Tensorflow ops for computing flow to image.
	"""
	with tf.variable_scope(name), tf.device('/cpu:0'):
		img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
		img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
		img = img / 127.5 - 1.
		return img


def hightlight_flow(flow):
	""" Convert flow into middlebury color code image.
	"""
	out = []
	s = flow.shape
	for i in range(flow.shape[0]):
		img = np.ones((s[1], s[2], 3)) * 144
		u = flow[i, :, :, 0]
		v = flow[i, :, :, 1]
		for h in range(s[1]):
			for w in range(s[1]):
				ui = u[h,w]
				vi = v[h,w]
				img[ui, vi, :] = 255.
		out.append(img)
	return np.float32(np.uint8(out))


def hightlight_flow_tf(flow, name='flow_to_image'):
	""" Tensorflow ops for hightlight flow.
	"""
	with tf.variable_scope(name), tf.device('/cpu:0'):
		img = tf.py_func(hightlight_flow, [flow], tf.float32, stateful=False)
		img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
		img = img / 127.5 - 1.
		return img


def image2edge(image):
	""" Convert image to edges.
	"""
	out = []
	for i in range(image.shape[0]):
		img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
		out.append(img)
	return np.float32(np.uint8(out))


class Encoder:
	# def __init__(self):
	# 	# super(Encoder, self).__init__()
	# 	self.encode()

	
	def aes_enc(self, input, output, password):
		with open(input, "rb") as fin:
			with open(output, "wb") as fout:
				pyAesCrypt.encryptStream(fin, fout, password, bufferSize)
		fin.close()

		# Steganographic image removed for security purpose after usage
		# remove(input)
	

	def genData(self, data):
		newData = []
		for i in data:
			newData.append(format(ord(i), '08b'))
		return newData

	def modPixel(self, pixel, data):
		datalist = self.genData(data)
		lendata = len(datalist)
		imdata = iter(pixel)

		for i in range(lendata):
			pixel = [value for value in imdata.__next__()[:3] +
										imdata.__next__()[:3] +
										imdata.__next__()[:3]]
			for j in range(0, 8):
				if (datalist[i][j] == '0') and (pixel[j]%2 != 0):
					if (pixel[j]%2 != 0):
						pixel[j] -= 1
				elif (datalist[i][j] == '1') and (pixel[j]%2 == 0):
					pixel[j] -= 1

			if (i == lendata - 1):
				if (pixel[-1] % 2 == 0):
					pixel[-1] -= 1
			else:
				if (pixel[-1] % 2 != 0):
					pixel[-1] -= 1

			pixel = tuple(pixel)
			yield pixel[0:3]
			yield pixel[3:6]
			yield pixel[6:9]

	def encode_enc(self, newimg, data):
		w = newimg.size[0]
		(x, y) = (0, 0)

		for pixel in self.modPixel(newimg.getdata(), data):
			newimg.putpixel((x, y), pixel)
			if (x == w-1):
				x = 0
				y += 1
			else:
				x += 1

	# def encode(self, img, x):
	# 	# img = input("Enter the image name for steganography:  ")
	# 	try:
	# 		image = Image.open(img, 'r')
	# 		# data = input("Enter the secret message for transmission:  ")
	# 		# if (len(data) == 0):
	# 		# 	raise ValueError('Message is blank!')

	# 		newimg = image.copy()
	# 		# encoding message inside image using steganography
	# 		# self.encode_enc(newimg, data)
	# 		self.encode_enc(newimg, x)
	# 		# saving the steganographed image
	# 		newimg.save('stegano_'+img)

			
	# 		# sha 512 hashing and aes encryption
	# 		# password = input('Enter the password for encryption:  ')
	# 		random_password = ''.join([random.choice(string.ascii_letters + string.digits) 
	# 									for n in range(32)])
	# 		password = hashlib.sha512(random_password.encode('utf-8')).hexdigest()

	# 		self.aes_enc('stegano_'+img,'message.enc', password)

	# 		return newimg
			
	# 	except FileNotFoundError:
	# 		print('Incorrect file name / File does not exist!')

	def tf_encode(self, x, message):
		"""
		LSB matching algorithm (+-1 embedding)

		Params:
			x: tf tensor shape (batch_size, width, height, channel)
			information: array with int bits
			stego: name of image with hidden message
		"""
		with tf.variable_scope('Stego'):
			# n, width, height, channel = tuple(map(int, x.get_shape()))
			n, width, height, channel = tuple(map(int, x.shape))
			# info = np.random.randint(0, 2, (n, 1638))	# 0.4 bpp
			info = np.random.randint(0, 2, (n, 1638))	# 0.4 bpp
			# mask = np.zeros(list(x.get_shape()))
			mask = np.zeros(list(x.shape))

			# print(x.get_shape())
			print(x.shape)
			print('Num of images: %s' % n)
			for img_idx in range(n):
				print(img_idx)

				for i, bit in enumerate(info[img_idx]):
					ind, jnd = i // width, i - width * (i // width)
					if tf.to_int32((x[img_idx, ind, jnd, 0] + 1) * 127.5) % 2 != bit:
						if np.random.randint(0, 2) == 0:
							# tf.assign_sub(x[img_idx, ind, jnd, 0], 1)
							mask[img_idx, ind, jnd, 0] += 1/256.
						else:
							# tf.assign_add(x[img_idx, ind, jnd, 0], 1)
							mask[img_idx, ind, jnd, 0] -= 1/256.

			# logger.debug('Finish encoding')
			return tf.add(x, mask)

	def tf_encode_no_batch(self, x, message):
		"""
		LSB matching algorithm (+-1 embedding)

		Params:
			x: tf tensor shape (batch_size, width, height, channel)
			information: array with int bits
			stego: name of image with hidden message
		"""
		with tf.variable_scope('Stego_no_batch'):
			width, height, channel = tuple(map(int, x.shape))
			info = np.random.randint(0, 2, (1, 1638))	# 0.4 bpp
			mask = np.zeros(list(x.shape))

			print(x.shape)
			for img_idx in range(1):
				print(img_idx)

				for i, bit in enumerate(info[img_idx]):
					ind, jnd = i // width, i - width * (i // width)
					if tf.to_int32((x[ind, jnd, 0] + 1) * 127.5) % 2 != bit:
						if np.random.randint(0, 2) == 0:
							# tf.assign_sub(x[img_idx, ind, jnd, 0], 1)
							mask[ind, jnd, 0] += 1/256.
						else:
							# tf.assign_add(x[img_idx, ind, jnd, 0], 1)
							mask[ind, jnd, 0] -= 1/256.

			# logger.debug('Finish encoding')
			return tf.add(x, mask)

	def encode(self, container, information, stego='stego.png'):
		"""
		LSB Matching algorithm (+-1 embedding)
		:param container: path to image container
		:param information: array with int bits
		:param stego: name of image with hidden message
		"""
		img = Image.open(container)
		width, height = img.size
		img_matr = np.asarray(img)
		img_matr.setflags(write=True)

		red_ch = img_matr[:, :, 0].reshape((1, -1))[0]

		information = np.append(information, np.ones(100, dtype=int))
		for i, bit in enumerate(information):
			if bit != red_ch[i] & i:
				if np.random.randint(0, 2) == 0:
					red_ch[i] -= 1
				else:
					red_ch[i] += 1
		img_matr[:, :, 0] = red_ch.reshape((height, width))

		Image.fromarray(img_matr).save(stego, "PNG")
		print("Finish encoding")

	def lsb_embed(self, input, message, mask):
		lsb = tf.to_int32((input+1) * 127.5) % 2
		pm1_mask = tf.to_float(tf.where(tf.equal(lsb, message),
										tf.zeros_like(mask),
										1 - 2*mask))
		pm1_mask = tf.stop_gradient(pm1_mask)
		embed = tf.add(input, pm1_mask / 127.5)

		return embed, pm1_mask, lsb


class Decoder:
	def __init__(self, password):
		# super(Decoder, self).__init__()
		self.password = password

		# sha 512 hashing
		password = hashlib.sha512(password.encode('utf-8')).hexdigest()

		# aes decryption and reverse steganography
		print(self.aes_dec('message.enc', password))

	
	def aes_dec(self, input, password):
		flag = True
		try:
			with open(input, "rb") as fin:
				with open("stegano_output.png", "wb") as fout:
					try:
						encFileSize = stat(input).st_size
						pyAesCrypt.decryptStream(fin, fout, password, bufferSize, encFileSize)
					except ValueError:
						print('Error: Incorrect password!')
						fout.close()
						flag = False
						# remove("stegano_output.png")

		except FileNotFoundError:
			flag = False
			print('Encrypted message file not found!')

		if flag:
			return self.decode()
		else:
			return 'Failed to decrypt!'
	

	def decode(self):
		image = Image.open('stegano_output.png', 'r')
		data = ''
		imgdata = iter(image.getdata())
		image.close()
		# remove("stegano_output.png")

		while (True):
			pixels = [value for value in imgdata.__next__()[:3] + imgdata.__next__()[:3] + imgdata.__next__()[:3]]
			binstr = ''

			for i in pixels[:8]:
				if (i % 2 == 0):
					binstr += '0'
				else:
					binstr += '1'

			data += chr(int(binstr, 2))
			if (pixels[-1]%2 != 0):
				return 'The decrypted message is:  ' + data


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
	parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
	parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
	args = parser.parse_args()
	test_contextual_attention(args)
