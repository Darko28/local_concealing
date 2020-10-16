""" Common model for DCGAN """
import logging

import random
import string
from PIL import Image

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from tensorflow.contrib.layers import fully_connected as linear
from tensorflow.contrib.layers import batch_norm as batch_norm

import neuralgym as ng
from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty, gan_log_loss
from neuralgym.ops.gan_ops import random_interpolates

from conceal_ops import gen_conv, gen_deconv, dis_conv
from conceal_ops import random_bbox, bbox2mask, local_patch
from conceal_ops import spatial_discounting_mask
from conceal_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class ConcealingModel(Model):

	def __init__(self, algo):
		super().__init__('ConcealingModel')
		self.algo = algo

	def build_conceal_net(self, x, mask, config=None, reuse=False,
						  training=True, padding='SAME', name='conceal_net'):
		""" Concealing network.

		Args:
			x: incomplete image, [-1, 1]
			mask: mask region {0, 1}
		Returns:
			[-1, 1] as predicted image
		"""
		xin = x
		offset_flow = None
		ones_x = tf.ones_like(x)[:, :, :, 0:1]
		x = tf.concat([x, ones_x, ones_x*mask], axis=3)

		# two stage network
		cnum = 32
		with tf.variable_scope(name, reuse=reuse), \
				arg_scope([gen_conv, gen_deconv],
						   training=training, padding=padding):
			# stage 1
			x = gen_conv(x, cnum, 5, 1, name='conv1')
			x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
			x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
			x = gen_conv(x, 4*cnum, 3, 2, name='conv3_downsample')
			x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
			x = gen_conv(x, 4*cnum, 3, 1, name='conv6')

			mask_s = resize_mask_like(mask, x)

			x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
			x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
			x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
			x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
			x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
			x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
			x = gen_deconv(x, 2*cnum, name='conv13_upsample')
			x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
			x = gen_deconv(x, cnum, name='conv15_upsample')
			x = gen_conv(x, cnum//2, 3, 1, name='conv16')
			x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
			x = tf.clip_by_value(x, -1., 1.)
			x_stage1 = x

			# return x_stage1, None, None,

			# stage 2, paste result as input
			# x = tf.stop_gradient(x)
			x = x*mask + xin*(1.-mask)
			x.set_shape(xin.get_shape().as_list())

			# conv branch
			xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
			x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
			x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
			x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
			x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
			x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
			x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
			x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
			x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
			x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
			x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
			x_hallu = x

			# attention branch
			x = gen_conv(xnow, cnum, 5, 1 , name='pmconv1')
			x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
			x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
			x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
			x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
			x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
						activation=tf.nn.relu)

			x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
			x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
			x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
			pm = x
			x = tf.concat([x_hallu, pm], axis=3)

			x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
			x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
			x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
			x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
			x = gen_deconv(x, cnum, name='allconv15_upsample')
			x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
			x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
			x_stage2 = tf.clip_by_value(x, -1., 1.)

		return x_stage1, x_stage2, offset_flow

	def build_wgan_local_discriminator(self, x, reuse=False, training=True):
		with tf.variable_scope('discriminator_local', reuse=reuse):
			cnum = 64
			x = dis_conv(x, cnum, name='conv1', training=training)
			x = dis_conv(x, cnum*2, name='conv2', training=training)
			x = dis_conv(x, cnum*4, name='conv3', training=training)
			x = dis_conv(x, cnum*8, name='conv4', training=training)
			x = flatten(x, name='flatten')
			return x

	def build_wgan_global_discriminator(self, x, reuse=False, training=True):
		with tf.variable_scope('discriminator_global', reuse=reuse):
			cnum = 64
			x = dis_conv(x, cnum, name='conv1', training=training)
			x = dis_conv(x, cnum*2, name='conv2', training=training)
			x = dis_conv(x, cnum*4, name='conv3', training=training)
			x = dis_conv(x, cnum*4, name='conv4', training=training)
			x = flatten(x, name='flatten')
			return x

	def build_wgan_discriminator(self, batch_local, batch_global,
								 reuse=False, training=True):
		with tf.variable_scope('discriminator', reuse=reuse):
			dlocal = self.build_wgan_local_discriminator(
				batch_local, reuse=reuse, training=training)
			dglobal = self.build_wgan_global_discriminator(
				batch_global, reuse=reuse, training=training)
			dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
			dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')

			print('discriminator dout_global shape: %s' % dout_global.get_shape())

			return dout_local, dout_global

	def steganalyzer_loss(real_output, generated_output):
		real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
		generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
		total_loss = real_loss + generated_loss
		return total_loss

	# @add_arg_scope
	def build_wgan_steganalyzer(self, x, config, reuse=False, training=True):
		with tf.variable_scope('discriminator', reuse=reuse):
			with tf.variable_scope('steganalyzer', reuse=reuse):
				x = self.preprocess_image(x)

				# cnum = 10			
				# x = tf.layers.conv2d(x, cnum, 5, padding='SAME', name='conv1', training=training)
				# x = tf.keras.layers.BatchNormalization(x)
				# x = tf.nn.leaky_relu(x)
				# x = 

				# x = tf.contrib.layers.batch_norm(x)
				x = tf.layers.conv2d(x, 10, 7, padding='SAME', name='conv1')
				# x = dis_conv(x, 10, 7, name='conv1', training=training)
				x = tf.contrib.layers.batch_norm(x)
				x = tf.nn.leaky_relu(x)
				# x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

				x = tf.contrib.layers.batch_norm(x)
				x = tf.layers.conv2d(x, 20, 5, padding='SAME', name='conv2')
				x = tf.contrib.layers.batch_norm(x)
				x = tf.nn.leaky_relu(x)
				x = tf.nn.max_pool(x, [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')

				x = tf.contrib.layers.batch_norm(x)
				x = tf.layers.conv2d(x, 30, 3, padding='SAME', name='conv3')
				x = tf.contrib.layers.batch_norm(x)
				x = tf.nn.leaky_relu(x)
				# x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

				x = tf.contrib.layers.batch_norm(x)
				x = tf.layers.conv2d(x, 40, 3, padding='SAME', name='conv4')
				x = tf.contrib.layers.batch_norm(x)
				x = tf.nn.leaky_relu(x)

				x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

				print(x.shape)

				# x = tf.reshape(x, [config.BATCH_SIZE, -1])

				print(x.shape)

				# x = flatten(x, name='flatten')
				# print(x.shape)


				x = linear(x, 100, activation_fn=tf.nn.tanh, scope='FC1')
				x = linear(x, 2, activation_fn=tf.nn.softmax, scope='OUT')

				x = flatten(x, name='flatten')

				x = tf.layers.dense(x, 1, name='steg_fc')


				# x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

				# x = tf.keras.layers.Conv2D(8, 5, padding='same', activation='relu', use_bias=False)(x)
				# x = tf.keras.layers.BatchNormalization(x)
				# x = tf.nn.leaky_relu(x)
				# x = tf.keras.layers.AveragePooling2D(pool_size=5, strides=2)(x)

				# x = tf.keras.layers.Conv2D(16, 5, padding='same', use_bias=False)(x)
				# x = tf.keras.layers.BatchNormalization(x)
				# x = tf.nn.tanh(x)
				# # x = tf.nn.leaky_relu(x)
				# x = tf.keras.layers.AveragePooling2D(5, 2)

				# x = tf.keras.layers.Conv2D(32, 1, padding='valid', use_bias=False)(x)
				# x = tf.keras.layers.BatchNormalization()
				# x = tf.nn.leaky_relu(x)
				# x = tf.keras.layers.AveragePooling2D(5, 2)

				# x = tf.keras.layers.Conv2D(64, 1, padding='valid', use_bias=False)(x)
				# x = tf.keras.layers.BatchNormalization(x)
				# x = tf.nn.leaky_relu(x)
				# x = tf.keras.layers.AveragePooling2D(5, 2)

				# x = tf.keras.layers.Conv2D(128, 1, padding='valid', use_bias=False)(x)
				# x = tf.keras.layers.BatchNormalization(x)
				# x = tf.nn.leaky_relu(x)

				# x = tf.keras.layers.Dense(128*(1+2*2+4*4))(x)
				# x = tf.nn.leaky_relu(x)
				# x = tf.keras.layers.Dense(128)(x)
				# x = tf.nn.tanh(x)
				# # x = tf.nn.leaky_relu(x)
				# x = tf.keras.activation.sigmoid(x)

				return x


	def preprocess_image(self, x):
		# obtain the secret embedding region (Object detection)
		# embed the secret message into each training imags
		K = 1 / 12. * tf.constant([
			[-1, 2, -2, 2, -1],
			[2, -6, 8, -6, 2],
			[-2, 8, -12, 8, -2],
			[2, -6, 8, -6, 2],
			[-1, 2, -2, 2, -1]
			], dtype=tf.float32)

		# kernel = tf.stack([K, K, K])
		# kernel = tf.stack([kernel, kernel, kernel])

		kernel = np.eye(3)[..., np.newaxis, np.newaxis] * K[np.newaxis, np.newaxis]

		return tf.nn.conv2d(x, tf.transpose(kernel, [2, 3, 0, 1]), [1, 1, 1, 1], padding='SAME')


	# def build_graph_with_losses(self, batch_data, bbox, encoder, config, training=True, 
	# 							summary=False, reuse=False):
	def build_graph_with_losses(self, batch_data, encoder, config, training=True, 
								summary=False, reuse=False):
		batch_pos = batch_data / 127.5 - 1.

		# self.algo = encoder

		# generate mask, 1 represents masked point
		bbox = random_bbox(config)
		mask = bbox2mask(bbox, config, name='mask_c')
		# mask = 
		batch_incomplete = batch_pos*(1.-mask)
		x1, x2, offset_flow = self.build_conceal_net(
			batch_incomplete, mask, config, reuse=reuse, training=training,
			padding=config.PADDING)
		if config.PRETRAIN_COARSE_NETWORK:
			batch_predicted = x1
			logger.info('Set batch_predicted to x1.')
		else:
			batch_predicted = x2
			logger.info('Set batch_predicted to x2.')
		losses = {}

		# apply mask and complete image
		batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
		# local patches
		local_patch_batch_pos = local_patch(batch_pos, bbox)
		local_patch_batch_predicted = local_patch(batch_predicted, bbox)
		local_patch_x1 = local_patch(x1, bbox)
		local_patch_x2 = local_patch(x2, bbox)
		local_patch_batch_complete = local_patch(batch_complete, bbox)
		local_patch_mask = local_patch(mask, bbox)
		l1_alpha = config.COARSE_L1_ALPHA
		losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1)*spatial_discounting_mask(config))
		if not config.PRETRAIN_COARSE_NETWORK:
			losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)*spatial_discounting_mask(config))
		losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
		if not config.PRETRAIN_COARSE_NETWORK:
			losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
		losses['ae_loss'] /= tf.reduce_mean(1.-mask)

		if summary:
			scalar_summary('losses/l1_loss', losses['l1_loss'])
			scalar_summary('losses/ae_loss', losses['ae_loss'])
			viz_img = [batch_pos, batch_incomplete, batch_complete]
			if offset_flow is not None:
				viz_img.append(
					resize(offset_flow, scale=4,
						   func=tf.image.resize_nearest_neighbor))
			images_summary(
				tf.concat(viz_img, axis=2),
				'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)


		# embed message
		message = ''.join([random.choice(string.ascii_letters + string.digits) 
							for n in range(32)])
		# with tf.Session() as sess:
		# print(tf.clip_by_value((batch_complete+1.)*127.5, 0, 255))
		print('batch_pos shape: %s' % batch_pos.get_shape())
		print('batch_complete: %s' % batch_complete[0])
		stego = self.algo.tf_encode(batch_complete, message=message)

		print('stego shape: %s' % stego.get_shape())
		# tmp = batch_complete[1,:,:,:]
		# tmp = tf.cast(tmp, tf.uint16)
		# tmp = tf.clip_by_value((tmp+1.)*127.5, 0, 255)
		# tmp = tf.cast(tmp, tf.uint16)
		# print(batch_complete)
		# with tf.Session().as_default():
		# 	tmp = batch_complete.eval()
		# 	tmp = Image.fromarray(tmp)
		# stego = self.algo.encode(tf.image.encode_png(batch_complete), message)

		# gan
		batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
		# local deterministic patch
		local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
		if config.GAN_WITH_MASK:
			batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
		# wgan with gradient penalty
		if config.GAN == 'wgan_gp':
			# seperate gan
			pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training=training, reuse=reuse)
			pos_local, neg_local = tf.split(pos_neg_local, 2)
			pos_global, neg_global = tf.split(pos_neg_global, 2)
			print('pos_global shape: %s' % pos_global.get_shape())
			print('pos_neg_global shape: %s' % pos_neg_global.get_shape())
			# wgan loss
			g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
			g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
			losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
			losses['d_loss'] = d_loss_global + d_loss_local
			# gp
			interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
			interpolates_global = random_interpolates(batch_pos, batch_complete)
			dout_local, dout_global = self.build_wgan_discriminator(
				interpolates_local, interpolates_global, reuse=True)

			print('interpolates_global shape: %s' % interpolates_global.get_shape())
			print('dout_global shape: %s' % dout_global.get_shape())
			print('batch_complete shape: %s' % batch_complete.get_shape())

			# steg_global = self.build_wgan_steganalyzer(batch_pos_neg, training=training, reuse=reuse)
			# pos_steg_global, neg_steg_global = tf.split(steg_global, 2)

			# g_loss_steg, d_loss_steg = gan_wgan_loss(batch_complete, stego, name='gan/steg_gan')
			# dout_steg = self.build_wgan_steganalyzer(batch_complete, reuse=True)
			steg_pos_neg = tf.concat([batch_complete, stego], axis=0)
			print('steg_pos_neg shape: %s' % steg_pos_neg.get_shape())
			dout_steg = self.build_wgan_steganalyzer(steg_pos_neg, config=config, reuse=reuse)
			pos_dout_steg, neg_dout_steg = tf.split(dout_steg, 2)
			# penalty_steg = gradients_penalty(interpolates_global, steg_global, mask=mask)
			# losses['d_loss'] = d_loss_global + d_loss_local + d_loss_steg
			print('dout_steg shape: %s' % dout_steg.get_shape())
			print('batch_pos_neg shape: %s' % batch_pos_neg.get_shape())
			print('steg_pos_neg shape: %s' % steg_pos_neg.get_shape())
			# steg
			# steg_g_loss, steg_d_loss = gan_log_loss(pos_dout_steg, neg_dout_steg, name='steg')
			steg_g_loss, steg_d_loss = gan_wgan_loss(pos_dout_steg, neg_dout_steg, name='gan/steg')
			losses['steg_d_loss'] = steg_d_loss
			interpolates_steg = random_interpolates(batch_complete, stego)
			print('interpolates_steg shape: %s' % interpolates_steg.get_shape())
			dout_stego = self.build_wgan_steganalyzer(interpolates_steg, config=config, reuse=True)
			penalty_steg = gradients_penalty(interpolates_steg, dout_stego)

			# apply penalty
			penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
			penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
			# losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
			losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global + penalty_steg)
			losses['d_loss'] = losses['d_loss'] + losses['gp_loss'] + losses['steg_d_loss']


			if summary and not config.PRETRAIN_COARSE_NETWORK:
				gradients_summary(g_loss_local, batch_predicted, name='g_loss_local')
				gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')

				gradients_summary(steg_g_loss, batch_predicted, name='steg_g_loss')
				scalar_summary('convergence/d_loss', losses['d_loss'])
				scalar_summary('convergence/local_d_loss', d_loss_local)
				scalar_summary('convergence/global_d_loss', d_loss_global)

				scalar_summary('convergence/steg_d_loss', losses['steg_d_loss'])
				# scalar_summary('gan_wgan_loss/gp_penalty_steg', penalty_steg)

				scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
				scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
				scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)
				scalar_summary('gan_wgan_loss/gp_penalty_steg', penalty_steg)

		if summary and not config.PRETRAIN_COARSE_NETWORK:
			# summary the magnitude of gradients from different losses w.r.t. predicted image
			gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
			gradients_summary(losses['g_loss'], x1, name='g_loss_to_x1')
			gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
			gradients_summary(losses['l1_loss'], x1, name='l1_loss_to_x1')
			gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
			gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
			gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
		if config.PRETRAIN_COARSE_NETWORK:
			losses['g_loss'] = 0
		else:
			losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
		losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
		logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
		logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
		if config.AE_LOSS:
			losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
			logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
		g_vars = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, 'conceal_net')
		# tf.add_to_collection('discriminator', 'steganalyzer')
		d_vars = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		return g_vars, d_vars, losses

	def build_infer_graph(self, batch_data, config, bbox=None, name='val'):
		"""
		"""
		config.MAX_DELTA_HEIGHT = 0
		config.MAX_DELTA_WIDTH = 0
		if bbox is None:
			bbox = random_bbox(config)
		mask = bbox2mask(bbox, config, name=name+'mask_c')
		batch_pos = batch_data / 127.5 - 1.
		edges = None
		batch_incomplete = batch_pos*(1.-mask)
		# concealing
		x1, x2, offset_flow = self.build_conceal_net(
			batch_incomplete, mask, config, reuse=True,
			training=False, padding=config.PADDING)
		if config.PRETRAIN_COARSE_NETWORK:
			batch_predicted = x1
			logger.info('Set batch_predicted to x1.')
		else:
			batch_predicted = x2
			logger.info('Set batch_predicted to x2.')
		# apply mask and reconstruct
		batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
		# global image visualization
		viz_img = [batch_pos, batch_incomplete, batch_complete]
		if offset_flow is not None:
			viz_img.append(
				resize(offset_flow, scale=4,
					   func=tf.image.resize_nearest_neighbor))
		images_summary(
			tf.concat(viz_img, axis=2),
			name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
		return batch_complete

	def build_static_infer_graph(self, batch_data, config, name):
		"""
		"""
		# generate mask, 1 represents masked point
		bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
				tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
		return self.build_infer_graph(batch_data, config, bbox, name)

	def build_server_graph(self, batch_data, reuse=False, is_training=False):
		"""
		"""
		# generate mask, 1 represents masked point
		batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
		masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

		batch_pos = batch_raw / 127.5 - 1.
		batch_incomplete = batch_pos * (1. - masks)

		# conceal
		x1, x2, flow = self.build_conceal_net(
			batch_incomplete, masks, reuse=reuse, training=is_training,
			config=None)
		batch_predict = x2

		# apply mask and reconstruct
		batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
		return batch_complete
