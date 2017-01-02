import numpy as np
import tensorflow as tf
import SMLP_MC
import SMLP_MC_train
import SMLP_input
import label_propagation
import random
import os



flags = tf.app.flags
FLAGS = flags.FLAGS
digits = SMLP_MC_train.digits


def labelsTomatrix(labels, digits):
	labelmatrix = np.zeros((len(digits), len(labels)))
	for i in xrange(0, len(labels)):
		labelmatrix[digits.index(labels[i])][i]=1
	return np.transpose(labelmatrix)

def placeholder_inputs():
	images_placeholder = tf.placeholder(tf.float32, shape=(None,
				FLAGS.width, FLAGS.height,FLAGS.dimension))
	keep_prob = tf.placeholder("float");
	return images_placeholder, keep_prob


def fill_feed_dict(dataset, digits, images_pl,keep_prob):
	(images, labels) = dataset.output_alldata()
	feed_dict = {
		images_pl: images,
		keep_prob: 1.0
	}
	return feed_dict,images,labels

'''
	The feed size of each class will be equal in this function
'''
def feed_equal_size_dict(images_placeholder,keep_prob, size=1):
	digitsdic = {}
	labelsdic = {}
	for i in digits:
		digitsdic[i],labelsdic[i]=SMLP_input.DataSet(dataset="testing",digits=[i]).next_batch(size)
	images = np.concatenate(digitsdic.values(), axis=0)
	labels = np.concatenate(labelsdic.values(), axis=0)
	feed_dict = {
		images_placeholder: images,
		keep_prob: 1.0,
	}
	return feed_dict, images, labels

def print_test_output():
	#Testing_data = SMLP_input.DataSet(dataset="training",digits=digits, train_percent=0.3, last_part = True)
	Testing_data = SMLP_input.DataSet(dataset="testing",digits=digits)
	if os.path.isfile('saver/unlabeled.txt'):
		os.remove('saver/unlabeled.txt')
	if os.path.isfile('saver/unlabeledEuclidean.txt'):
		os.remove('saver/unlabeledEuclidean.txt')
	with tf.Graph().as_default():
		images_pl, keep_prob= placeholder_inputs()
		logits = SMLP_MC.inference(images_pl, keep_prob)
		saver = tf.train.Saver()
		sess = tf.Session()
		saver.restore(sess, "saver/"+str(FLAGS.train_percent)+"model.ckpt")
		feed_dict,images,labels=fill_feed_dict(Testing_data, digits, images_pl,keep_prob)
		logits_value = sess.run(logits, feed_dict=feed_dict)
		size = logits_value.shape[0]
		with open('saver/unlabeled.txt', 'a') as f:
			for i in xrange(0, size):
				f.write(str(labels[i])+"\t"+str(logits_value[i,:])+"\n")
		with open('saver/unlabeledEuclidean.txt', 'a') as f:
			for i in xrange(0, size):
				f.write(str(labels[i])+"\t"+str(np.concatenate(images[i,:,:,0]))+"\n")

def print_train_output():
	Training_data = SMLP_input.DataSet(dataset="training",digits=digits, train_percent=FLAGS.train_percent)
	if os.path.isfile('saver/labeled.txt'):
		os.remove('saver/labeled.txt')
	if os.path.isfile('saver/labeledEuclidean.txt'):
		os.remove('saver/labeledEuclidean.txt')
	with tf.Graph().as_default():
		images_pl, keep_prob= placeholder_inputs()
		logits = SMLP_MC.inference(images_pl, keep_prob)
		saver = tf.train.Saver()
		sess = tf.Session()
		saver.restore(sess, "saver/"+str(FLAGS.train_percent)+"model.ckpt")
		'''
		Using random size class to feed
		'''
		#feed_dict,images,labels=fill_feed_dict(Training_data, digits, images_pl,keep_prob)
		'''
		Using fixed size class to feed
		'''
		feed_dict,images,labels=feed_equal_size_dict(images_pl,keep_prob, size=FLAGS.fixed_size)

		logits_value = sess.run(logits, feed_dict=feed_dict)
		size = logits_value.shape[0]
		with open('saver/labeled.txt', 'a') as f:
			for i in xrange(0, size):
				f.write(str(labels[i])+"\t"+str(logits_value[i,:])+"\n")
		with open('saver/labeledEuclidean.txt', 'a') as f:
			for i in xrange(0, size):
				f.write(str(labels[i])+"\t"+str(np.concatenate(images[i,:,:,0]))+"\n")

def main():
	print "convariance matrix"
	SMLP_MC_train.print_convariance_matrix()
	print "test_output: This step will take lots of time please be patient"
	print_test_output()
	print "train_output: This step will also take lots of time, Thank you for your patience"
	print_train_output()
	print 'Our accuracy:'
	print label_propagation.compute_accuracy('saver/labeled.txt', 'saver/unlabeled.txt')
	print 'Euclidean accuracy:'
	print label_propagation.compute_accuracy('saver/labeledEuclidean.txt', 'saver/unlabeledEuclidean.txt')

if __name__=='__main__':
	main()