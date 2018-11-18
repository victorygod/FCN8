import tensorflow as tf
import numpy as np
import os
from data import Dataloader

activation = tf.nn.relu

class FCN:
	def __init__(self, istraining = True):
		self.num_class = 21
		self.input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])
		self.label_tensor = tf.placeholder(tf.float32, [None, None, None, self.num_class])
		self.logits = self.build(istraining)
		self.output = tf.nn.softmax(self.logits)
		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.label_tensor)
		self.loss = tf.reduce_mean(self.loss)
		self.saver = tf.train.Saver(max_to_keep=1)

	def build(self, istraining = True):
		vgg_trainable = True
		VGG_MEAN = [103.939, 116.779, 123.68]
		red, green, blue = tf.split(self.input_tensor, 3, 3)
		bgr = tf.concat([
		                blue - VGG_MEAN[0],
		                green - VGG_MEAN[1],
		                red - VGG_MEAN[2],
		            ], 3)
		conv1_1 = tf.layers.conv2d(bgr, 64, 3, name = "conv1_1", padding = "same", activation = activation, trainable = vgg_trainable)
		conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, name = "conv1_2", padding = "same", activation = activation, trainable = vgg_trainable)
		pool1 = tf.layers.max_pooling2d(conv1_2, pool_size = 2, strides = 2, padding = "same")

		conv2_1 = tf.layers.conv2d(pool1, 128, 3, name = "conv2_1", padding = "same", activation = activation, trainable = vgg_trainable)
		conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, name = "conv2_2", padding = "same", activation = activation, trainable = vgg_trainable)
		pool2 = tf.layers.max_pooling2d(conv2_2, pool_size = 2, strides = 2, padding = "same")

		conv3_1 = tf.layers.conv2d(pool2, 256, 3, name = "conv3_1", padding = "same", activation = activation, trainable = vgg_trainable)
		conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, name = "conv3_2", padding = "same", activation = activation, trainable = vgg_trainable)
		conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, name = "conv3_3", padding = "same", activation = activation, trainable = vgg_trainable)
		pool3 = tf.layers.max_pooling2d(conv3_3, pool_size = 2, strides = 2, padding = "same")

		conv4_1 = tf.layers.conv2d(pool3, 512, 3, name = "conv4_1", padding = "same", activation = activation, trainable = vgg_trainable)
		conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, name = "conv4_2", padding = "same", activation = activation, trainable = vgg_trainable)
		conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, name = "conv4_3", padding = "same", activation = activation, trainable = vgg_trainable)
		pool4 = tf.layers.max_pooling2d(conv4_3, pool_size = 2, strides = 2, padding = "same")

		conv5_1 = tf.layers.conv2d(pool4, 512, 3, name = "conv5_1", padding = "same", activation = activation, trainable = vgg_trainable)
		conv5_2 = tf.layers.conv2d(conv5_1, 512, 3, name = "conv5_2", padding = "same", activation = activation, trainable = vgg_trainable)
		conv5_3 = tf.layers.conv2d(conv5_2, 512, 3, name = "conv5_3", padding = "same", activation = activation, trainable = vgg_trainable)
		pool5 = tf.layers.max_pooling2d(conv5_3, pool_size = 3, strides = 2, padding = "same")

		fc6 = tf.layers.conv2d(pool5, 4096, 7, name = "fc6", padding = "same", activation = activation, trainable = vgg_trainable)
		fc6 = tf.layers.dropout(fc6, training = istraining)
		fc7 = tf.layers.conv2d(fc6, 4096, 1, name = "fc7", padding = "same", activation = activation, trainable = vgg_trainable)
		fc7 = tf.layers.dropout(fc7, training = istraining)

		score_fr = tf.layers.conv2d(fc7, self.num_class, 1, name = "score_fr")

		# shape1 = tf.shape(pool4)
		# upscore2 = tf.image.resize_images(score_fr, (shape1[1], shape1[2]), method = tf.image.ResizeMethod.BILINEAR)
		# upscore2 = tf.layers.conv2d(upscore2, self.num_class, 4, padding = "same", name = "upscore2")
		# score_pool4 = tf.layers.conv2d(pool4, self.num_class, 1, padding = "same", name = "score_pool4", kernel_initializer=tf.zeros_initializer())
		# fuse_pool4 = tf.add(upscore2, score_pool4)

		# shape2 = tf.shape(pool3)
		# upscore4 = tf.image.resize_images(fuse_pool4, (shape2[1], shape2[2]), method = tf.image.ResizeMethod.BILINEAR)
		# upscore4 = tf.layers.conv2d(upscore4, self.num_class, 4, padding = "same", name = "upscore4")
		# score_pool3 = tf.layers.conv2d(pool3, self.num_class, 1, padding = "same", name = "score_pool3", kernel_initializer=tf.zeros_initializer())
		# fuse_pool3 = tf.add(upscore4, score_pool3)

		# shape3 = tf.shape(conv1_1)
		# upscore32 = tf.image.resize_images(fuse_pool3, (shape3[1], shape3[2]), method = tf.image.ResizeMethod.BILINEAR)
		# upscore32 = tf.layers.conv2d(upscore32, self.num_class, 4, padding = "same", name = "upscore32")


		score_pool4 = tf.layers.conv2d(pool4, self.num_class, 1, padding = "same", name = "score_pool4", kernel_initializer=tf.zeros_initializer())
		shape1 = tf.shape(score_pool4)
		kernel_shape = [4, 4, self.num_class, self.num_class]
		upscore2 = self.deconv_layer("upscore2", score_fr, shape1, kernel_shape, 2)
		fuse_pool4 = tf.add(upscore2, score_pool4)

		score_pool3 = tf.layers.conv2d(pool3, self.num_class, 1, padding = "same", name = "score_pool3", kernel_initializer=tf.zeros_initializer())
		shape2 = tf.shape(score_pool3)
		kernel_shape = [4, 4, self.num_class, self.num_class]
		upscore4 = self.deconv_layer("upscore4", fuse_pool4, shape2, kernel_shape, 2)
		fuse_pool3 = tf.add(upscore4, score_pool3)

		shape3 = tf.shape(conv1_1)
		kernel_shape = [16,16,self.num_class, self.num_class]
		upscore32 = self.deconv_layer("upscore32", fuse_pool3, [shape3[0],shape3[1],shape3[2], self.num_class], kernel_shape, 8, is_zero_init = True)

		return upscore32

	def deconv_layer(self, name, input_layer, output_shape, kernel_shape, stride, is_zero_init = False):
		with tf.variable_scope(name) as scope:
			if is_zero_init:
				kernel = tf.get_variable(name = "kernel", initializer = tf.zeros_initializer(), shape = kernel_shape)
			else:
				kernel = self.get_deconv_kernel(kernel_shape)
		deconv = tf.nn.conv2d_transpose(input_layer, kernel, output_shape, strides = [1,stride,stride,1], padding = "SAME")
		return deconv

	def get_deconv_kernel(self, shape):
		kernel = np.zeros(shape)
		width = shape[0]
		height = shape[1]
		f = np.ceil(width/2.0)
		c = (2 * f - 1 - f % 2) / (2.0 * f)
		bilinear = np.zeros((shape[0], shape[1]))
		for x in range(width):
		    for y in range(height):
		        bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
		for i in range(shape[2]):
		    kernel[:,:,i,i] = bilinear

		init = tf.constant_initializer(value=kernel, dtype=tf.float32)
		return tf.get_variable(name = "kernel", initializer = init, shape = shape)

	def train(self):
		self.train_step = tf.train.AdamOptimizer(3e-4).minimize(self.loss)
		dataloader = Dataloader()
		i = 0
		c = tf.ConfigProto()
		c.gpu_options.allow_growth = True
		with tf.Session(config = c) as sess:
			sess.run(tf.global_variables_initializer())
			checkpoint = tf.train.latest_checkpoint("checkpoint/")
			if checkpoint:
				print("restore from: " + checkpoint)
				self.saver.restore(sess, checkpoint)
			elif os.path.exists("vgg16.npy"):
				print("restore from vgg weights.")
				vgg = np.load("vgg16.npy", encoding='latin1').item()
				ops = []
				vgg_dict = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3",
				"conv5_1","conv5_2","conv5_3", "fc6", "fc7"]
				tf_variables = {}
				for variables in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
					if "Adam" or "RMS" in variables.name: continue
					key = variables.name.split("/")[0].split(":")[0]
					if key not in vgg_dict: continue
					if key not in tf_variables:
						tf_variables[key] = [variables]
						ops.append(variables.assign(vgg[key][0]))
					else:
						tf_variables[key].append(variables)
						ops.append(variables.assign(vgg[key][1]))
				sess.run(ops)
			for inputs, labels in dataloader.generate():
				_, lo, preds = sess.run([self.train_step, self.loss, self.output], feed_dict={self.input_tensor: inputs, self.label_tensor: labels})
				print(i, lo)
				if i%20==0:
					dataloader.save_images("output.jpg", preds)
					dataloader.save_images("label.jpg", labels)
				i+=1
				if i%100==99:
					self.saver.save(sess, "checkpoint/ckpt")

	def eval(self):
		dataloader = Dataloader()
		c = tf.ConfigProto()
		c.gpu_options.allow_growth = True
		with tf.Session(config = c) as sess:
			checkpoint = tf.train.latest_checkpoint("checkpoint/")
			if checkpoint:
				print("restore from: " + checkpoint)
				self.saver.restore(sess, checkpoint)
			nn=0
			for inputs in dataloader.get_images():
				preds = sess.run(self.output, feed_dict = {self.input_tensor:inputs})
				dataloader.save_images(str(nn)+".jpg", preds)
				nn+=1
net = FCN()
# net.eval()
net.train()