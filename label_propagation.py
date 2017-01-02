import numpy as np
from sklearn import semi_supervised


SHUFFLE_DATA = True
np.random.seed(10715)

# Outputs of our neural network.
OUR_TRAIN_FILEPATH = 'saver/labeled.txt'
OUR_TEST_FILEPATH = 'saver/unlabeled.txt'

# For comparing our accuracy to accuracy when label propagation is used
# on the images themselves (vector of pixel values).
IMAGE_TRAIN_FILEPATH = 'saver/imagetrain.txt'
IMAGE_TEST_FILEPATH = 'saver/imagetest.txt'

def parse_input_file(filepath):
  X = []
  y = []
  with open(filepath, 'r') as f:
    for line in f:
      line_parts = line.split('\t', 1)
      # Hack to remove starting '[' and ending ']\n' so that
      # np.fromstring actually works right.
      X.append(np.fromstring(line_parts[1][1: -2], sep=' '))
      y.append(int(line_parts[0]))
  X = np.array(X)
  y = np.array(y)
  if not SHUFFLE_DATA:
    return X, y

  perm = np.random.permutation(len(X))
  return X[perm], y[perm]

def compute_accuracy(train_filepath, test_filepath):
  X_train, y_train = parse_input_file(train_filepath)
  X_test, y_test = parse_input_file(test_filepath)

  X = np.concatenate((X_train, X_test))
  # Use test data as unlabeld data (denoted by -1).
  y = np.concatenate((y_train, [-1 for _ in X_test]))

  # TODO(bparr): Experiment between LabelPropagation and LabelSpreading and
  #     the different initialization arguments for each.
  label_prop_model = semi_supervised.LabelPropagation()
  #label_prop_model = semi_supervised.LabelSpreading()
  label_prop_model.fit(X, y)

  # Strangely, the fit() changes the label of some of the training data,
  # despite using the "Hard Clamp" mode (alpha = 1). So avoid any issue by
  # simply extracting the propagated test labels.
  # TODO(bparr): Figure out why hard clamping is performing unexpectedly.
  y_test_predicted = label_prop_model.transduction_[y_train.shape[0]:]

  # Print confusion matrix where the rows are actual classes and
  # the columns are the predicted classes.
  L = len(set(y_train))
  confusion_matrix = [[0 for _ in xrange(L)] for __ in xrange(L)]
  for i in xrange(y_test.shape[0]):
    confusion_matrix[y_test[i]][y_test_predicted[i]] += 1
    #if y_test[i] != y_test_predicted[i]:
    #  print i,y_test[i],y_test_predicted[i]
  for confusion_line in confusion_matrix:
    print ','.join(str(x) for x in confusion_line)

  correct = np.sum(y_test_predicted == y_test)
  return 1.0 * correct / len(y_test)

def main():
  #print 'Image pixel accuracy:'
  #print compute_accuracy(IMAGE_TRAIN_FILEPATH, IMAGE_TEST_FILEPATH)
  #print ''
  print 'Our accuracy:'
  print compute_accuracy(OUR_TRAIN_FILEPATH, OUR_TEST_FILEPATH)


if __name__ == '__main__':
  main()