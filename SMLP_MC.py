import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name = 'weights')

def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name = 'bias')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

def inference(images, keep_prob):
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.tanh(conv2d(images, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

	#Second Layer
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

	#Third Layer fully connected
	with tf.name_scope('fully_connected'):
		W_fc1 = weight_variable([FLAGS.width/4 * FLAGS.height/4 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, FLAGS.width/4 * FLAGS.height/4 * 64])
		h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#Dropout
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	#Forth layer
	with tf.name_scope('output'):
		W_fc2 = weight_variable([1024, FLAGS.out_size])
		b_fc2 = bias_variable([FLAGS.out_size])
		logits=tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	return logits

def loss(logits, labels, global_step = 0):
	# labels.shape = (batch_size, label_size)
	# logits.shape = (batch_size, size of output Layer)
	#Y^T * F
	#YTF = tf.matmul(tf.transpose(labels) , logits)
	# Y^T * F * F^T * Y
	#weight = tf.matmul( YTF, tf.transpose(YTF))

	'''
	average of innerproduct squre
	'''
	#G = tf.matmul(logits, tf.transpose(logits))
	#Gsquare = tf.square(G)
	#weight = tf.matmul(tf.matmul(tf.transpose(labels), Gsquare),labels)
	#A = tf.reduce_sum(tf.square(tf.reduce_sum(labels, 0)))
	#B = tf.sub(tf.square(tf.to_float(labels.get_shape()[0])), A)
	#T = tf.div(tf.trace(weight), A)
	#S = tf.div(tf.sub(tf.reduce_sum(weight),tf.trace(weight)), B)
	#loss = tf.sub(S,tf.div(T, 5.0))
	#return loss

	'''
	cross-entropy loss
	'''
	#lambda_value = (tf.to_float(labels.get_shape()[1]) -1.0)*tf.to_float(global_step)/float(FLAGS.max_steps)+1.0
	G = tf.matmul(logits, tf.transpose(logits))
	Gsquare_one = tf.div(tf.square(G),tf.to_float(logits.get_shape()[1]*logits.get_shape()[1]))
	Gsquare_two = tf.sub(tf.ones(tf.shape(Gsquare_one),tf.float32),tf.identity(Gsquare_one))
	y_one = tf.matmul(labels, tf.transpose(labels))
	y_two = tf.sub(tf.ones(tf.shape(y_one),tf.float32),y_one)
	A = tf.reduce_sum(tf.square(tf.reduce_sum(labels, 0)))
	B = tf.sub(tf.square(tf.reduce_sum(labels)), A)
	#loss = -tf.div(tf.reduce_sum(tf.mul(y_one, tf.log(Gsquare_one+1e-5))),A)-lambda_value*tf.div(tf.reduce_sum(tf.mul(y_two, tf.log(Gsquare_two+1e-5))),B)
	loss = -tf.div(tf.reduce_sum(tf.mul(y_one, tf.log(Gsquare_one+1e-5))),A)-5.0*tf.div(tf.reduce_sum(tf.mul(y_two, tf.log(Gsquare_two+1e-5))),B)
	return loss

def cross_entropy_loss(logits,labels):
	# labels.shape = (batch_size, label_size)
	# logits.shape = (batch_size, size of output Layer)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
	loss = tf.reduce_mean(cross_entropy)
	return loss


def training(loss, starter_learning_rate, global_step = 0 ):
	if global_step==0:
		learning_rate = starter_learning_rate
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_op = optimizer.minimize(loss)
	else:
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
									100, 0.9, staircase = True)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,beta2=0.999,epsilon=1e-8, use_locking=False)
		train_op = optimizer.minimize(loss, global_step = global_step)
	return train_op