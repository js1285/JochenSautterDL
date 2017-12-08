import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#
#
# A CONVOLUTIONAL AUTOENCODER FOR MNIST
#
#


#
#
# DEFINE THE NETWORK
#
#

# helper function to define tf variables for initialized layer weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# helper function to define tf variables for initialized layer bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# helper function to define convolution layers
# noinspection PyShadowingNames
def conv2d(x, W):
    # same convolution stride 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# helper function to define convolution layers
# noinspection PyShadowingNames
def trans_conv2d(x, W, outputShape):
    return tf.nn.conv2d_transpose(x, W, output_shape=outputShape,
                                  strides=[1, 2, 2, 1], padding='SAME')


# helper function to define pooling layers
# noinspection PyShadowingNames
def max_pool_2x2(x):
    # max pooling with 2x2 blocks
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


#
# now realy start defining the convolutional network
#

# specify input values
# (given later as placeholder)
x = tf.placeholder(tf.float32, shape=[None, 784])
# reshape network input (placeholder) to 4D tensor
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 1st layer (conv-relu-maxpool)
# 3x3 filter with 1 input channels and 8 output channels
# bias for each output channel
# reduces size to 14x14
W_conv1 = weight_variable([3, 3, 1, 8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 2nd layer (conv-relu-maxpool)
# 3x3 filter with 8 input channels and 4 output channels
# reduces size to 7x7
W_conv2 = weight_variable([3, 3, 8, 4])
b_conv2 = bias_variable([4])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 3rd layer (conv-relu)
# 3x3 filter with 4 input channels and 2 output channels
# reduces size to 7x7
W_conv3 = weight_variable([3, 3, 4, 2])
b_conv3 = bias_variable([2])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)


# 4th layer (transconv-relu)
# 2x2 transpose-filter with 2 input channels and 4 output channels
# increases size to 14x14
W_tconv4 = weight_variable([2, 2, 4, 2])  # [x, y, outshape, inshape]
b_tconv4 = bias_variable([4])
h_tconv4 = tf.nn.relu(trans_conv2d(h_conv3, W_tconv4, outputShape=[64, 14, 14, 4]) + b_tconv4)


# 5th layer (conv-relu)
# 3x3 filter with 4 input channels and 4 output channels
W_conv5 = weight_variable([3, 3, 4, 4])
b_conv5 = bias_variable([4])
h_conv5 = tf.nn.relu(conv2d(h_tconv4, W_conv5) + b_conv5)


# 6th layer (transconv-relu)
# 2x2 transpose-filter with 4 input channels and 8 output channels
# increases size to 28x28
W_tconv6 = weight_variable([2, 2, 8, 4])  # [x, y, outshape, inshape]
b_tconv6 = bias_variable([8])
h_tconv6 = tf.nn.relu(trans_conv2d(h_conv5, W_tconv6, outputShape=[64, 28, 28, 8]) + b_tconv6)


# 7th layer (conv-relu)
# 3x3 filter with 8 input channels and 8 output channels
W_conv7 = weight_variable([3, 3, 8, 8])
b_conv7 = bias_variable([8])
h_conv7 = tf.nn.relu(conv2d(h_tconv6, W_conv7) + b_conv7)


# 8th layer (conv-relu)
# 1x1 filter with 8 input channels and 1 output channels
W_conv8 = weight_variable([1, 1, 8, 1])
b_conv8 = bias_variable([1])
h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)


# squared loss
mean_squared_loss = tf.reduce_mean(tf.squared_difference(h_conv8, x_image))


#
#
# TRAINING
#
#

# define training step with ADAM
trainsteps = [(tf.train.AdamOptimizer(0.1).minimize(mean_squared_loss), 0.1),
              (tf.train.AdamOptimizer(0.01).minimize(mean_squared_loss), 0.01),
              (tf.train.AdamOptimizer(0.001).minimize(mean_squared_loss), 0.001)]

# load the training data set MNIST
mnist = input_data.read_data_sets('MNIST_data')  # optional one_hot=True (not necessary here)

# run training:
# logging train and validation loss every 100 iterations
with tf.Session() as sess:
    for trainstep, learningRate in trainsteps:
        print("start training autoencoder with learning rate", learningRate)
        sess.run(tf.global_variables_initializer())

        #
        # train autoencoder with certain learning rate
        #
        for i in range(15000):
            batch = mnist.train.next_batch(64)
            if i % 100 == 0:
                #
                # calculate training and validation loss here
                #
                trainloss = mean_squared_loss.eval(feed_dict={
                    x: batch[0]})
                print('step {}, train loss (one batch a 64) {}'.format(i, trainloss))
                validationloss = mean_squared_loss.eval(feed_dict={
                    x: mnist.test.next_batch(64)[0]})
                print('step {}, validation loss (one val.batch a 64) {}'.format(i, validationloss))
            # train step on same batch already used for logging. train step should happen after logging in this case.
            trainstep.run(feed_dict={x: batch[0]})
        print("finished training autoencoder")

        #
        # print some examples
        #
        example_train_immages = mnist.train.next_batch(64)[0]
        example_train_processed = h_conv8.eval(feed_dict={x: example_train_immages})
        example_test_immages = mnist.test.next_batch(64)[0]
        example_test_processed = h_conv8.eval(feed_dict={x: example_test_immages})
        print("EXAMPLE RESULTS")

        print("EXAMPLE TRAIN IMMAGES")
        for imnr in range(len(example_train_immages)):
            print("IMNR", imnr)
            for pixnr in range(len(example_train_immages[0])):
                print(example_train_immages[imnr][pixnr])
        print("EXAMPLE TRAIN PROCESSED")
        for imnr in range(len(example_train_processed)):
            print("IMNR", imnr)
            for pixnr_X in range(len(example_train_processed[0])):
                for pixnr_Y in range(len(example_train_processed[0][0])):
                    print(example_train_processed[imnr][pixnr_X][pixnr_Y])
        print("EXAMPLE TEST IMMAGES")
        for imnr in range(len(example_test_immages)):
            print("IMNR", imnr)
            for pixnr in range(len(example_test_immages[0])):
                print(example_test_immages[imnr][pixnr])
        print("EXAMPLE TEST PROCESSED")
        for imnr in range(len(example_test_processed)):
            print("IMNR", imnr)
            for pixnr_X in range(len(example_test_processed[0])):
                for pixnr_Y in range(len(example_test_processed[0][0])):
                    print(example_test_processed[imnr][pixnr_X][pixnr_Y])


print("program finished")
