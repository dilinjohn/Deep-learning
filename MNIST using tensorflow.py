import tensorflow as tf
print('Tensor flow version is ', tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random

def TRAIN_SIZE(num):
	print('Total training examples in Dataset =' + str(mnist.train.images.shape[0]))
	print('Training sample shape =' + str(mnist.train.images.shape))
	X_train = mnist.train.images[:num, :]
	y_train = mnist.train.labels[:num, :]
	print('X_train examples loaded =' + str(X_train.shape))
	print('y_train labels loaded =' + str(y_train.shape))
	print('')
	return X_train, y_train

def TEST_SIZE(num):
	print('Total test examples in Dataset =' + str(mnist.test.images.shape[0]))
	print('Test sample shape =' + str(mnist.test.images.shape))
	X_test = mnist.test.images[:num, :]
	y_test = mnist.test.labels[:num, :]
	print('X_test exmaples loaded =' + str(X_test.shape))
	print('y_test labels loaded =' + str(y_test.shape))
	print('')
	return X_test, y_test

# And we’ll define some simple functions for resizing and displaying the data:

#y_train = mnist.train.labels[:1000, :]
#print(y_train[500])
def display_digit(num):
	print(y_train[num])
	label = y_train[num].argmax(axis=0)
	image = X_train[num].reshape([28, 28])
	plt.title('Example: %d  Label: %d' % (num, label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()

# Flatten multiple images
def display_mult_flat(start, stop): # flatten the image
	images = X_train[start].reshape([1, 784])
	for i in range(start+1, stop):
		images = np.concatenate((images, X_train[i].reshape(1, 784)))
	plt.imshow(images, cmap= plt.get_cmap('gray_r'))
	plt.show()


# Build and train the model
#X_train, y_train = TRAIN_SIZE(55000)

# Display digits at random
#display_digit(np.random.randint(0, X_train.shape[0]))
#display_mult_flat(0,400)

import tensorflow as tf
sess = tf.Session()

# define placeholder - A placeholder, as the name suggests, is a variable used to feed data into.
# When we assign None to our placeholder, it means the placeholder can be fed as many examples as you want to give it
X = tf.placeholder(tf.float32, shape=[None, 784])

# define y_, which will be used to feed y_train into.
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # used later to comapre ground truth to predictions

# define weights and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the classifier function
y= tf.nn.softmax(tf.matmul(X, W) + b)

# Run a session whereby we will feed some data to the placeholders
# Feed our classifier 3 examples and see the predictions
# Some tests

X_train, y_train = TRAIN_SIZE(3)
# first intitalize our variables
sess.run(tf.global_variables_initializer())
print(sess.run(y, feed_dict={X:X_train}))

'''
here we can see our prediction for our first three training examples. 
Of course, our classifier knows nothing at this point, so it outputs an equal 10% probability
of our training examples for each possible class.
'''

# Loss function or Cost function - goal is to minimize the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_  * tf.log(y), reduction_indices=[1]))


X_train, y_train = TRAIN_SIZE(5500)
X_test, y_test = TEST_SIZE(10000)
LEARNING_RATE = 0.1
TRAIN_STEPS = 2500

# Initialize the variables so that they can be used for the tensorflow graph

init = tf.global_variables_initializer()
sess.run(init)

# Train the classifier using Gradient Descent
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_predition = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))

# Now, we’ll define a loop that repeats TRAIN_STEPS times; for each loop, it runs training, 
# feeding in values from x_train and y_train using feed_dict. 

for i in range(TRAIN_STEPS+1):
	sess.run(training, feed_dict={X:X_train, y_:y_train})
	if i%100 == 0:
		print('Training step ' + str(i) +' Accuracy ' + str(sess.run(accuracy, feed_dict={X:X_test, y_:y_test})) + '  Loss = '
			+ str(sess.run(cross_entropy, feed_dict={X:X_train, y_:y_train})))



# Visualize the weights ' W'
for i in range(10):
	plt.subplot(2,5, i+1)
	weight = sess.run(W)[:, i]
	plt.title(i)
	plt.imshow(weight.reshape([28, 28]), cmap=plt.get_cmap('seismic'))
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_visible(False)  # hide the whole x-axis
	frame1.axes.get_yaxis().set_visible(False)  # hide the whole y-axis

plt.show()




