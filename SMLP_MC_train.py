import numpy as np
import tensorflow as tf
import SMLP_input
import SMLP_MC
import random
import math
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('number_of_epoch', 10, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must be even.')
flags.DEFINE_integer('width', 28, 'Width of image')
flags.DEFINE_integer('height', 28, 'Height of image')
flags.DEFINE_integer('dimension', 1, 'Dimension of image')
flags.DEFINE_float('train_percent', 0.01, 'Size of training dataset')
flags.DEFINE_integer('out_size', 10, 'Size of last layer')
flags.DEFINE_integer('fixed_size', 5, 'Size of each class')
digits = [0,1,2,3,4,5,6,7,8,9]
#max_steps =int(math.ceil((FLAGS.number_of_epoch*60000*FLAGS.train_percent)/FLAGS.batch_size*len(digits)/10))
max_steps=1000

def placeholder_inputs(label_size):
	images_placeholder = tf.placeholder(tf.float32, shape=(None,
				FLAGS.width, FLAGS.height,FLAGS.dimension))
	labels_placeholder = tf.placeholder(tf.float32, shape=(None,label_size))
	keep_prob = tf.placeholder("float");
	return images_placeholder, labels_placeholder, keep_prob

def convariance_placeholder_inputs():
	images_placeholder = tf.placeholder(tf.float32, shape=(None,
				FLAGS.width, FLAGS.height,FLAGS.dimension))
	keep_prob = tf.placeholder("float");
	return images_placeholder, keep_prob

def fill_feed_dict(dataset, digits, images_pl,labels_pl,keep_prob,dropout):
	(images_feed, labels) = dataset.next_batch(FLAGS.batch_size)
	labels_feed = labelsTomatrix(labels, digits)
	feed_dict = {
		images_pl: images_feed,
		labels_pl: labels_feed,
		keep_prob: dropout,
	}
	return feed_dict

def labelsTomatrix(labels, digits):
	labelmatrix = np.zeros((len(digits), len(labels)))
	for i in xrange(0, len(labels)):
		labelmatrix[digits.index(labels[i])][i]=1
	return np.transpose(labelmatrix)


def do_eval(sess, loss, digits, dataset, images_pl, labels_pl, keep_prob):
	(images_feed, labels) = dataset.output_alldata()
	labels_feed = labelsTomatrix(labels, digits)
	feed_dict = {
		images_pl: images_feed,
		labels_pl: labels_feed,
		keep_prob: 1.0
	}
	loss_value = sess.run(loss, feed_dict=feed_dict)
	print '\tTest Loss:'+ str(loss_value)

def feed_convariance_dict(images_pl,keep_prob):
	np.set_printoptions(threshold=np.NaN)
	np.set_printoptions(linewidth=np.NaN)
	np.set_printoptions(precision=2)
	digitsdic = {}
	for i in digits:
		digitsdic[i],_=SMLP_input.DataSet(dataset="testing",digits=[i]).next_batch(1)
	images = np.concatenate(digitsdic.values(), axis=0)
	feed_dict = {
		images_pl: images,
		keep_prob: 1.0,
	}
	return feed_dict

'''
	The feed size of each class will be equal in this function
'''
def feed_equal_size_dict(size=1):
	digitsdic = {}
	labelsdic = {}
	for i in digits:
		digitsdic[i],labelsdic[i]=SMLP_input.DataSet(dataset="testing",digits=[i]).next_batch(size)
	images = np.concatenate(digitsdic.values(), axis=0)
	labels = labelsTomatrix(np.concatenate(labelsdic.values(), axis=0),digits)
	return images, labels

def print_convariance_matrix():
	with tf.Graph().as_default():
		images_placeholder, keep_prob= convariance_placeholder_inputs()
		logits = SMLP_MC.inference(images_placeholder, keep_prob)
		saver = tf.train.Saver()
		sess = tf.Session()
		saver.restore(sess, "saver/"+str(FLAGS.train_percent)+"model.ckpt")
		feed_dict = feed_convariance_dict(images_placeholder,keep_prob)
		logits_value = sess.run(logits, feed_dict=feed_dict)
		size = len(digits)
		covariance = np.empty([size, size])
		for i in xrange(0,size):
			for j in xrange(0,size):
				covariance[i][j]=np.inner(logits_value[i,:],logits_value[j,:])
		print covariance



def run_training():
	Training_data = SMLP_input.DataSet(dataset="training",digits=digits, train_percent=FLAGS.train_percent)
	Testing_data = SMLP_input.DataSet(dataset="testing",digits=digits)
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)
		images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(len(digits))
		logits = SMLP_MC.inference(images_placeholder, keep_prob)
		loss = SMLP_MC.loss(logits, labels_placeholder, global_step)
		train_op = SMLP_MC.training(loss, FLAGS.learning_rate, global_step)
		sess = tf.Session()
		saver = tf.train.Saver()
		try:
			saver.restore(sess, "saver/amodel.ckpt")
		except:
			init = tf.initialize_all_variables()
			sess.run(init)

		images, labels = feed_equal_size_dict(FLAGS.fixed_size) #using in fixed size

		for step in xrange(0,max_steps):
			'''
			Using random size class to train
			'''
			#feed_dict = fill_feed_dict(Training_data, digits,
			#			images_placeholder, labels_placeholder,keep_prob, 0.5)
			'''
			Using equal size class to train
			'''
			feed_dict = {
				images_placeholder: images,
				labels_placeholder: labels,
				keep_prob: 1.0,
			}
			#feed_dict =feed_equal_size_dict(images_placeholder, labels_placeholder,keep_prob, size=FLAGS.fixed_size)

			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			if step %100 ==0:
				print "Step:"+str(step)+"\tTrain_loss:"+str(loss_value)
				'''
				Calculate the loss function in whole test dataset
				'''
				#do_eval(sess, loss ,digits, Testing_data, images_placeholder, labels_placeholder,keep_prob)
			if step == max_steps-1:
				save_path = saver.save(sess, "saver/"+str(FLAGS.train_percent)+"model.ckpt")
				print "Final Step:"+str(step)+"\tTrain_loss:"+str(loss_value)
				print "Model saved in file: ", save_path

def supervised_training():
	Training_data = SMLP_input.DataSet(dataset="training",digits=digits, train_percent=FLAGS.train_percent)
	Testing_data = SMLP_input.DataSet(dataset="testing",digits=digits)
	with tf.Graph().as_default():
		'''
			training
		'''
		global_step = tf.Variable(0, trainable=False)
		images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(len(digits))
		logits = SMLP_MC.inference(images_placeholder, keep_prob)
		loss = SMLP_MC.cross_entropy_loss(logits, labels_placeholder)
		train_op = SMLP_MC.training(loss, FLAGS.learning_rate, global_step)
		sess = tf.Session()
		saver = tf.train.Saver()
		init = tf.initialize_all_variables()
		sess.run(init)

		images, labels = feed_equal_size_dict(FLAGS.fixed_size)  #using fixed size

		for step in xrange(0,max_steps):
			'''
			Using random size class to train
			'''
			#feed_dict = fill_feed_dict(Training_data, digits,
			#			images_placeholder, labels_placeholder,keep_prob, 0.5)
			'''
			Using equal size class to train
			'''
			feed_dict = {
				images_placeholder: images,
				labels_placeholder: labels,
				keep_prob: 1.0,
			}
			#feed_dict =feed_equal_size_dict(images_placeholder, labels_placeholder,keep_prob, size=FLAGS.fixed_size)

			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			if step%100 ==0:
				print "Step:"+str(step)+"\tTrain_loss:"+str(loss_value)
		'''
			testing
		'''
		(images_feed, labels) = Testing_data.output_alldata()
		labels_feed = labelsTomatrix(labels, digits)
		feed_dict = {
			images_placeholder: images_feed,
			labels_placeholder: labels_feed,
			keep_prob: 1.0
		}
		correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(labels_placeholder,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
		accuracy_value = sess.run(accuracy, feed_dict=feed_dict)
		print "The accuracy of supervised cross_entropy_loss accuracy is:"+str(accuracy_value)
		print "If you want to compare our method and Euclidean distance, please run SMLP_MC_output.py"



def main():
	print 'Training process of our method: kernel learning'
	run_training()
	print 'The convariance matrix of our method'
	print_convariance_matrix()
	print 'The training process of cross-entropy loss LeNet supervised learning'
	supervised_training()



if __name__=='__main__':
	main()