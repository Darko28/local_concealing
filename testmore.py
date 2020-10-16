import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

import time

from concealing_model_normal_gan import ConcealingModel
from conceal_ops import Encoder

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":

    start = time.clock()

    ng.get_gpus(1)
    args = parser.parse_args()
    encoder = Encoder()
    model = ConcealingModel(encoder)
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:

        time1 = time.clock() 

        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(input_image)
# config = ng.Config('inpaint.yml')
# output = model.build_infer_graph(input_image, config=config)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(args.output, result[0][:, :, ::-1])

        duration = (time.clock() - time1)
        print("Session time used:", duration)

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
