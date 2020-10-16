import argparse

import cv2
import numpy as np
import tensorflow as tf

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import neuralgym as ng

import matplotlib.pyplot as plt

from concealing_model_normal_gan import ConcealingModel
from conceal_ops import Encoder


parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
# 					help='The filename of image to be completed.')
# parser.add_argument('--mask', default='', type=str,
# 					help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='output.png', type=str,
# 					help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='server_model/20190109170929693687_amax_imagenet_NORMAL_wgan_gp_full_model_coco/', type=str,
					help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
	ng.get_gpus(1)
	config = ng.Config('conceal.yml')
	args = parser.parse_args()

	encoder = Encoder()

	ssim_arr = []
	psnr_arr = []

	model = ConcealingModel(encoder)
	# image = cv2.imread(args.image)
	# global_mask = cv2.imread(args.mask)
	global_mask = cv2.imread('./center_mask_256.png')

	# assert image.shape == global_mask.shape

	with open(config.DATA_FLIST[config.DATASET][2]) as f:
		fnames = f.read().splitlines()
	# data = ng.data.DataFromFNames(fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
	# images = data.data_pipeline(config.BATCH_SIZE)
	print("data: ", len(fnames))

	for i in range(0, len(fnames)):
		fname = fnames[i]
		image = cv2.imread(fname)
		image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		print("image shape: ", image.shape)
		print("mask shape: ", global_mask.shape)

		assert image.shape == global_mask.shape

		h, w, _ = image.shape
		grid = 8
		image = image[:h//grid*grid, :w//grid*grid, :]
		mask = global_mask[:h//grid*grid, :w//grid*grid, :]
		print('Shape of image: {}'.format(image.shape))
		print('Shape of mask: {}'.format(mask.shape))

		image = np.expand_dims(image, 0)
		mask = np.expand_dims(mask, 0)
		input_image = np.concatenate([image, mask], axis=2)

		sess_config = tf.ConfigProto()
		sess_config.gpu_options.allow_growth = True
		with tf.Session(config=sess_config) as sess:
			input_image = tf.constant(input_image, dtype=tf.float32)
			# output = model.build_server_graph(input_image)
			output = model.build_server_graph(input_image)

			print("output: ", output.get_shape())
			# stego = encoder.tf_encode_no_batch(output, 'nuist')

			# cover = tf.image.convert_image_dtype(output, tf.float32)
			# stego = tf.image.convert_image_dtype(stego, tf.float32)
			# ssim = sess.run(tf.image.ssim(cover, stego, max_val=1.0))
			# psnr = sess.run(tf.image.psnr(cover, stego, max_val=1.0))
			# print("ssim: ", ssim)
			# print("psnr: ", psnr)

			output = (output + 1.) * 127.5
			output = tf.reverse(output, [-1])
			output = tf.saturate_cast(output, tf.uint8)
			# load pretrained model
			vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			print("vars_list: ", len(vars_list))
			assign_ops = []
			for var in vars_list:
				vname = var.name
				from_name = vname
				var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
				assign_ops.append(tf.assign(var, var_value))
			sess.run(assign_ops)
			print('Model loaded.')
			result = sess.run(output)
			cv2.imwrite('./{}.png'.format(f"cover_{i}"), result[0][:, :, ::-1])

			cover = result[0][:,:,::-1]
			cover_rgb = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
			cv2.imwrite('./{}.png'.format(f"rgb_{i}"), cover_rgb)
			# cover = np.expand_dims(cover, axis=0)
			# stego = tf.reshape(stego, [256, 256, 3])
			# stego = encoder.tf_encode_no_batch(cover, 'nuist')
			stego = encoder.encode('./cover_{}.png'.format(f"{i}"), '111', stego='./stego_{}.png'.format(f"{i}"))
			# stego = encoder.tf_encode(cover, 'nuist')
			# sess.run(stego)

			# print("cover:", cover.shape)
			# print("stego: ", stego.shape)
			# cover = args.output
			# encoder.encode(cover, 'nuist')
			cover_img = cv2.imread('./cover_{}.png'.format(f"{i}"))
			stego_img = cv2.imread('./stego_{}.png'.format(f"{i}"))
			psnr_value = psnr(cover_img, stego_img)
			ssim_value = ssim(cover_img, stego_img, multichannel=True)
			# cover = tf.image.decode_image('/Users/darko/Desktop/local_concealing/cover_0.png')
			# stego = tf.image.decode_image('/Users/darko/Desktop/local_concealing/cover_0.png')
			# # cover = tf.image.convert_image_dtype(cover * 255, tf.float32)
			# # print("cover: ", cover.eval())
			# # print("stego: ", stego.eval())
			# # stego = tf.image.convert_image_dtype(stego, tf.float32)
			# # print("stego1: ", stego.eval())
			# ssim = sess.run(tf.image.ssim(cover, stego, max_val=255.0))
			# psnr = sess.run(tf.image.psnr(cover, stego, max_val=255.0))
			print("ssim: ", ssim_value)
			print("psnr: ", psnr_value)

			ssim_arr.append(ssim_value)
			psnr_arr.append(psnr_value)
			print("ssim_arr length: {}".format(len(ssim_arr)))

		if i%100 == 0:
			print("Drawing charts...")
			plt.plot(psnr_arr)
			plt.title('PSNR') 
			plt.xlabel('samples')
			plt.ylabel('psnr')
			# plt.ylim(ymin=0)
			plt.ylim((0.0, 70.0))
			plt.savefig('psnr_chart.png', dpi=300)

			plt.plot(ssim_arr)
			plt.title('SSIM') 
			plt.xlabel('samples')
			plt.ylabel('ssim')
			plt.ylim((0.0, 1.0))
			plt.savefig('ssim_chart.png', dpi=300)


	plt.plot(psnr_arr)
	plt.title('PSNR') 
	plt.xlabel('samples')
	plt.ylabel('psnr')
	# plt.ylim(ymin=0)
	plt.ylim((0.0, 70.0))
	plt.savefig('psnr_chart.png', dpi=300)

	plt.plot(ssim_arr)
	plt.title('SSIM') 
	plt.xlabel('samples')
	plt.ylabel('ssim')
	plt.ylim((0.0, 1.0))
	plt.savefig('ssim_chart.png', dpi=300)
	# plt.show()