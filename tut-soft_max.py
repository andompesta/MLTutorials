import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# functions for create positive weight
def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# functions for create positive bias
def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)



class LogRegression(object):
    def __init__(self, input, n_in, n_out):
        '''
        Initialize the parameters of the logistic regression

        :type input: tensorflow.tensor
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        '''

        self.input = input

        self.W = weight_variable([n_in, n_out], name='weight_l1_regression')
        self.b = bias_variable([n_out], name='bias_l1_regression')


        self.score = tf.matmul(self.input, self.W) + self.b                    # compute model score
        self.p_y_given_x = tf.nn.softmax(self.score)                            # compute the probability of each class for the input x
        self.y_pred = tf.argmax(self.p_y_given_x, 1)                            # get the class with highest probability

    def loss_funtion(self, y_true):
        '''
        cross entropy as loss function
        :param y_true: real labels of the input data
        :return: cross entropy function
        '''

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(self.p_y_given_x), reduction_indices=[1]))    # reduction_indices=[1] adds the elements in the second dimension of y
        return cross_entropy










if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)           # load mnist dataset



    n_in = 28 * 28                                                          # num input feature
    n_out = 10                                                              # num hidden layer


    x = tf.placeholder(tf.float32, shape=[None, n_in])                      # input variable
    y_true = tf.placeholder(tf.float32, shape=[None, n_out])                # true labels


    model = LogRegression(x, n_in, n_out)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(model.loss_funtion(y_true))
    correct_prediction = tf.equal(model.y_pred, tf.argmax(y_true,1))     # number of correct predicted images
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_true:batch_ys}, session=sess)    # eval performance every 100 iteration
            print("step %d, training accuracy %g"%(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

    print('\n--------------------------------------\nTEST PERFORMANCE\n')
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_true: mnist.test.labels}, session=sess))
