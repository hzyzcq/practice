import requests
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os


#用于处理的图像的属性
width = 299
height = 299
channels = 3 

INCEPTION_V3_CHECKPOINT_PATH = os.path.join("datasets", "inception","inception_v3.ckpt")    

#通过ImageNet的数据库下载不同种类的熊的图像URL并爬取
f = open('datasets/Images/panda.rtf', 'r') 
pandas = f.readlines()  #Urls of panda images

for i in range(66,100):
	with open('datasets/Images/panda/panda_' + str(i + 1) + '.jpg', 'wb') as f: 
		img = requests.get(pandas[9 + i]).content 
		f.write(img) 

for i in range(100,150):
	with open('datasets/Images/panda_test/panda_' + str(i + 1) + '.jpg','wb') as f:
		img = requests.get(pandas[9 + i]).content
		f.write(img)

#省略其他三类的爬取过程



#读取训练集和测试集的图像目录
BEAR_PATH = os.path.join('datasets', 'Images/train')     
bear_classes = sorted([dirname for dirname  in os.listdir(BEAR_PATH)
 if os.path.isdir(os.path.join(BEAR_PATH, dirname))])   

from collections import defaultdict

train_image_paths = defaultdict(list)
test_image_paths  = defaultdict(list)

for bear_class in bear_classes:
	image_dir = os.path.join(BEAR_PATH, bear_class)
	for filepath in os.listdir(image_dir):
		if filepath.endswith('.jpg'):
			train_image_paths[bear_class].append(os.path.join(image_dir, filepath))


for paths in train_image_paths.values():
	paths.sort()

BEAR_PATH = os.path.join('datasets', 'Images/test')  
for bear_class in bear_classes:
	image_dir = os.path.join(BEAR_PATH, bear_class)
	for file_path in os.listdir(image_dir):
		if file_path.endswith('.jpg'):
			test_image_paths[bear_class].append(os.path.join(image_dir, file_path))

for paths in test_image_paths.values():
	paths.sort()

#对图像进行标注以用于Softmax

bear_class_ids = {bear_class : index for index, bear_class in enumerate(bear_classes)}
train_paths = []
test_paths = []
for bear_class, paths in train_image_paths.items():
	for path in paths:
		train_paths.append((path, bear_class_ids[bear_class]))
np.random.shuffle(train_paths)

for bear_class, paths in test_image_paths.items():
	for path in paths:
		test_paths.append((path, bear_class_ids[bear_class]))
np.random.shuffle(test_paths)


#批处理
from random import sample

def random_batch(paths, batch_size):
	batch_paths = sample(paths, batch_size)
	images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths]
	prepared_images = [prepare_image(image) for image in images]
	X_batch = 2 * np.stack(prepared_images) - 1
	y_batch = np.array([labels for path, labels in batch_paths], dtype = np.int32)
	return X_batch, y_batch



import matplotlib.image as mpimg

from scipy.misc import imresize

def prepare_image(image, target_width = 299, target_height = 299, max_zoom = .2):
	height = image.shape[0]
	width = image.shape[1]
	image_ratio = width / height
	target_image_ratio = target_width / target_height
	crop_vertically = image_ratio < target_image_ratio
	crop_width = width if crop_vertically else int(height * target_image_ratio)
	crop_height = int(width / target_image_ratio) if crop_vertically else height

	resize_factor = np.random.rand() * max_zoom + 1.0
	crop_width = int(crop_width / resize_factor)
	crop_height = int(crop_height / resize_factor)

	x0 = np.random.randint(0, width - crop_width)
	y0 = np.random.randint(0, height - crop_height)
	x1 = x0 + crop_width
	y1 = y0 + crop_height

	image = image[y0 : y1, x0 : x1]

	if np.random.rand() < .5:
		image = np.fliplr(image)

	image = imresize(image, (target_width, target_height))

	return image.astype(np.float32) / 255


from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim


tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape = [None, height, width, channels], name = 'X')
training = tf.placeholder_with_default(False, shape = [])
with slim.arg_scope(inception.inception_v3_arg_scope()):
	logits, end_points = inception.inception_v3(X, num_classes = 1001, is_training = training)

saver = tf.train.Saver()

prelogits = tf.squeeze(end_points['PreLogits'], axis = [1,2])
n_outputs = len(bear_classes)

with tf.name_scope('new_output'):
	bear_logits = tf.layers.dense(prelogits, n_outputs, name = 'bear_logits')
	y_proba = tf.nn.softmax(bear_logits, name = 'Y_proba')

y = tf.placeholder(tf.int32, shape = [None])

training_rate = .01

with tf.name_scope('train'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = bear_logits, 
		labels = y)
	loss = tf.reduce_mean(xentropy)
	optimizer = tf.train.AdamOptimizer(training_rate)
	bear_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'bear_logits')
	training_op = optimizer.minimize(loss, var_list = bear_vars)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(bear_logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

n_epochs = 10
batch_size = 20
n_iterations_per_epoch = len(train_paths) // batch_size

with tf.Session() as sess:
	init.run()
	saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
	for epoch in range(n_epochs):
		print('Epoch', epoch, end = '')
		for iteration in range(n_iterations_per_epoch):
			print('.', end = '')
			X_batch, y_batch = random_batch(train_paths, batch_size)
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch, training:True})
		acc_batch = accuracu.eval(feed_dict = {X:X_batch, y:y_batch})
		print('  Last batch accuracy:', acc_batch)

		save_path = saver.save(sess, './bear_model')

X_test, y_test = random_batch(test_paths, batch_size=len(test_paths))
n_test_batches = 20
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session () as sess:
	saver.restore(sess, './bear_model')
	print('Evaluating test accuracy...')
	acc_test = np.mean([
		accuracy.eval(feed_dict {X : X_test_batch, y : y_test_batch})
		for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
	print('Test accuracy:', acc_test)



