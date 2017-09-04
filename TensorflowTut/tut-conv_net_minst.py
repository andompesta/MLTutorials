import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# functions for create positive weight
def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

# functions for create positive bias
def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

# function to apply convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# function to apply max-poling of 2X2 patches
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ConvLayer(object):
    def __init__(self, input, dims, name=''):
        self.input = input

        self.W_conv = weight_variable(dims, name='weight_'+name)                                      # filter of 5X5 with 1 input channels and 32 output channels (output filters)
        self.b_conv = bias_variable([dims[-1]], name='bias_'+name)                                  # bias of the first layer

        self.h_conv = tf.nn.relu(conv2d(self.input, self.W_conv) + self.b_conv)             # apply convolution between the filters and the images always a 28x28 dimentions
        self.output = max_pool_2x2(self.h_conv)                                             # max-poll the filtered images to highlight the borders and reduce the dimentions 14x14

class DensLayer(object):
    def __init__(self, input, dims, name=''):
        '''
        Constructor of a fully connected layer
        :param input: placeholder of the input
        :param dims: dimentions
        :param name: name of the layer
        '''
        self.input = input
        self.W = weight_variable(dims, name='weight_'+name)                            # weight for a fully-connected layer (input 7x7x64 values, output 1024 values)
        self.b = bias_variable([dims[-1]], name='bias_'+name)

        self.output = tf.nn.relu(tf.matmul(self.input, self.W) + self.b)              # apply the fully connected transformation


class ConvNet(object):
    def __init__(self, input, n_in, conv_dim, dens_dim, n_out, keep_prob):
        '''
        constructor of a CNN
        :param input: placeholder of the input
        :param n_in: input dimension
        :param n_hidden: hidden layers
        :param n_out: output_dimentions
        :param dropout_prob: dropout probability
        '''

        self.input = tf.reshape(input, [-1, n_in, n_in, 1])                         # reshape the input to the original image size (28X28 with 1 color channel)
        self.conv_layers = []
        self.dens_layers = []
        self.keep_prob = keep_prob

        #CONV_LAYERS
        for i in range(len(conv_dim)):
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.conv_layers[i - 1].output


            conv_layer = ConvLayer(layer_input, conv_dim[i], name='conv_layer_'+str(i))
            self.conv_layers.append(conv_layer)


        # DENSE LAYER
        for i in range(len(dens_dim)):
            if i == 0:
                layer_input = tf.reshape(self.conv_layers[-1].output, [-1, dens_dim[i][0]])             # flat conv putput
            else:
                layer_input = self.dens_layers[i-1].output

            dens_layer = DensLayer(layer_input, dens_dim[i], name='dens_layer_'+str(i))                 # create dens layer
            self.dens_layers.append(dens_layer)

        # DROPOUT
        self.h_fc1_drop = tf.nn.dropout(self.dens_layers[-1].output, keep_prob)                                           # apply dropout at the fully-connected layer

        # OUTPUT LAYER (softmax)
        self.W_fc2 = weight_variable([dens_dim[-1][-1], n_out], name='weight_softmax_layer')
        self.b_fc2 = bias_variable([n_out], name='bias_softmax_layer')

        self.score = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2                           # FINAL SCORE FUNCTION
        self.y_pred = tf.argmax(self.score, 1)

    def loss_function(self, y_true):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=self.score))   # LOSS FUNCTION

if __name__ == '__main__':


    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)           # load mnist dataset

    x = tf.placeholder(tf.float32, shape=[None, 784])                       # input variable
    y_true = tf.placeholder(tf.float32, shape=[None, 10])                       # true label
    keep_prob = tf.placeholder(tf.float32)                                  # dropout prob

    # n_in = 28
    # conv_dim = [[5, 5, 1, 32], [5, 5, 32, 64]]
    # dens_dim = [[7 * 7 * 64, 1024]]
    # n_out = 10

    # model = ConvNet(x, n_in, conv_dim, dens_dim, n_out, keep_prob)
    #
    # sess = tf.Session()
    #
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss_function(y_true))       # Optimizer to use
    # correct_prediction = tf.equal(model.y_pred, tf.argmax(y_true, 1))     # number of correct predicted images
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #
    # sess.run(tf.global_variables_initializer())
    #
    #
    # for i in range(20000):
    #     batch = mnist.train.next_batch(50)                                  # produce the next batch of size 50
    #     if i%100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_true: batch[1], keep_prob: 1.0}, session=sess)    # eval performance every 100 iteration
    #         print("step %d, training accuracy %g"%(i, train_accuracy))
    #     sess.run(train_step, {x: batch[0], y_true: batch[1], keep_prob: 0.5})                                         # execute one step of SGD
    #
    # print('\n--------------------------------------\nTEST PERFORMANCE\n')
    # print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1.0}, session=sess))


    x_image = tf.reshape(x, [-1, 28, 28,1])                                   # reshape the input to the original image size (28X28 with 1 color channel)


    # FIRST LAYER
    W_conv1 = weight_variable([5, 5, 1, 32])                                # filter of 5X5 with 1 input channels and 32 output channels (output filters)
    b_conv1 = bias_variable([32])                                           # bias of the first layer

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)                # apply convolution between the filters and the images always a 28x28 dimentions
    h_pool1 = max_pool_2x2(h_conv1)                                         # max-poll the filtered images to highlight the borders and reduce the dimentions 14x14

    # SECOND LAYER
    W_conv2 = weight_variable([5, 5, 32, 64])                               # produce 64 filters form the 32 initial
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)                                         # after max-poll we have 7x7 images (64 7x7 images)


    # DENSE LAYER
    W_fc1 = weight_variable([7 * 7 * 64, 1024])                             # weight for a fully-connected layer (input 7x7x64 values, output 1024 values)
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])                        # flatten the conv_layer output
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)              # apply the fully connected transformation

    # DROPOUT
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                            # apply dropout at the fully-connected layer


    # OUTPUT LAYER (softmax)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2                           # FINAL SCORE FUNCTION


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))   # LOSS FUNCTION

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)       # Optimizer to use
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))     # number of correct predicted images
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(50)                                  # produce the next batch of size 50
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)    # eval performance every 100 iteration
            print("step %d, training accuracy %g"%(i, train_accuracy))
        sess.run(train_step, {x: batch[0], y_: batch[1], keep_prob: 0.5})                                         # execute one step of SGD


    print('\n--------------------------------------\nTEST PERFORMANCE\n')
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, session=sess))




