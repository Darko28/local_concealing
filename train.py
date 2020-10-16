import os
import glob
import socket
import logging

from PIL import Image

import tensorflow as tf
import neuralgym as ng

from concealing_model_normal_gan import ConcealingModel
# from yolo3.yolo_video import detect_coco
# from yolo3.yolo import *
# from yolo3.yolo import *
# from yolo3 import *
# from yolo3 import yolo
from conceal_ops import Encoder

logger = logging.getLogger()


def multigpu_graph_def(model, data, config, gpu_id=0, loss_type='g'):
	with tf.device('/cpu:0'):
		images = data.data_pipeline(config.BATCH_SIZE)
	if gpu_id == 0 and loss_type == 'g':
		_, _, losses = model.build_graph_with_losses(
			images, encoder, config, summary=True, reuse=True)
	else:
		_, _, losses = model.build_graph_with_losses(
			batch_data=images, encoder=encoder, config=config, reuse=True)
	if loss_type == 'g':
		return losses['g_loss']
	elif loss_type == 'd':
		return losses['d_loss']
	else:
		raise ValueError('loss type is not supported.')


if __name__ == "__main__":

	# with tf.Session() as sess:
	# sess = tf.Session()
	config = ng.Config('conceal.yml')
	# if config.GPU_ID != -1:
	# 	ng.set_gpus(config.GPU_ID)
	# else:
	# 	ng.get_gpus(config.NUM_GPUS)

# yolo = YOLO()
	# training data
	with open(config.DATA_FLIST[config.DATASET][0]) as f:
		fnames = f.read().splitlines()
	data = ng.data.DataFromFNames(
		fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
	images = data.data_pipeline(config.BATCH_SIZE)

	print('fnames: %s' % fnames)
	# result = detect_coco(yolo, fnames[0])
	# print(result)
	print('fnames length: %s' % len(fnames))
	batch_result = []
	# for fname in fnames:
	# 	result = detect_coco(yolo, fname)
	# 	batch_result.append(result)
	# 	print('batch_result: %s' % batch_result)
	print('total batch_result: %s' % batch_result)

	encoder = Encoder()
	# encoder.encode()
	# password = input('Enter the password:  ')
	# decoder = Decoder(password)
	# # decoder.aes_dec()
	# print('Decrypt Finished!')

	# main model
	model = ConcealingModel(encoder)
	g_vars, d_vars, losses = model.build_graph_with_losses(
		images, encoder=encoder, config=config)
	# with tf.Session() as sess:
	# 	sess.run(tf.initialize_all_variables())
	# 	complete_img = batch_complete.eval(session=sess)

	# validation images
	if config.VAL:
		with open(config.DATA_FLIST[config.DATASET][1]) as f:
			val_fnames = f.read().splitlines()
		# progress monitor by visualizing static images
		for i in range(config.STATIC_VIEW_SIZE):
			static_fnames = val_fnames[i:i+1]
			static_images = ng.data.DataFromFNames(
				static_fnames, config.IMG_SHAPES, nthreads=1,
				random_crop=config.RANDOM_CROP).data_pipeline(1)
			static_concealed_images = model.build_static_infer_graph(
				static_images, config, name='static_view/%d' % i)

	# training settings
	lr = tf.get_variable(
		'lr', shape=[], trainable=False,
		initializer=tf.constant_initializer(1e-4))
	d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
	g_optimizer = d_optimizer

	# gradient processor
	if config.GRADIENT_CLIP:
		gradient_processor = lambda grad_var: (
			tf.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
			grad_var[1])
	else:
		gradient_processor = None

	# log dir
	log_prefix = 'model_logs/' + '_'.join(
		[ng.date_uid(), socket.gethostname(), config.DATASET,
		'MASKED' if config.GAN_WITH_MASK else 'NORMAL',
		config.GAN, config.LOG_DIR])

	# train discriminator with secondary trainer, should initialize before
	# primary trainer.
	discriminator_training_callback = ng.callbacks.SecondaryTrainer(
		pstep=1,
		optimizer=d_optimizer,
		var_list=d_vars,
		max_iters=5,
		graph_def=multigpu_graph_def,
		graph_def_kwargs={
			'model': model, 'data': data, 'config': config, 'loss_type': 'd'},
	)
	# train generator with primary trainer
	trainer = ng.train.Trainer(
		optimizer=g_optimizer,
		var_list=g_vars,
		max_iters=config.MAX_ITERS,
		graph_def=multigpu_graph_def,
		grads_summary=config.GRADS_SUMMARY,
		gradient_processor=gradient_processor,
		graph_def_kwargs={
			'model': model, 'data': data, 'config': config, 'loss_type': 'g'
		},
		spe=config.TRAIN_SPE,
		log_dir=log_prefix,
	)

	# add all callbacks
	if not config.PRETRAIN_COARSE_NETWORK:
		trainer.add_callbacks(discriminator_training_callback)
	trainer.add_callbacks([
		ng.callbacks.WeightsViewer(),
		ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
		ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
		ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.summary.merge_all()),
	])

	# launch training
	trainer.train()
