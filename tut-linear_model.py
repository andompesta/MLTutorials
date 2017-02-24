import tensorflow as tf
import numpy as np






W = tf.Variable([.15, .15], tf.float32, name='weight')
W = tf.reshape(W, [2, 1])

b = tf.Variable([-.3], tf.float32, name='bias')
x = tf.placeholder(tf.float32, shape=[None, 2])                     # placeholder for the 2D input
y = tf.placeholder(tf.float32, shape=[None, 1])                     # placheolder for the 1D output


linear_model = tf.matmul(x, W) + b                                  # define the linear model
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)                                # loss function

sess = tf.Session()                             # init the session
init = tf.global_variables_initializer()        # init the variables variable
sess.run(init)


# -------------PREPARE INPUT OF THE MODEL-------------
x_train = np.array([1., 1., 2., 2., 3., 3., 4., 4.])
x_train = x_train.reshape(4, 2)

y_train = np.array([0., -1.,-2.,-3.])
y_train = y_train.reshape([4, 1])

# # -------------RANDOM OUTPUT-------------
# print(sess.run(loss, {x:x_train, y:y_train}))       # output of a random initialization
#
#
# # -------------LEARNING THE WEIGHT-------------
# lr = tf.constant(0.01, tf.float32)                      # define learning rate
# optimizer = tf.train.GradientDescentOptimizer(lr)       # declare the optimizer to use
# train = optimizer.minimize(loss)                        # compute the gradient to minimize the loss function
#
#
# sess.run(init)                                          # reset values to incorrect defaults.
# for i in range(1000):
#     sess.run(train, {x:x_train, y:y_train})
#
# # evaluate training accuracy
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# -------------USING TENSORFLOW LIBRARY-------------

# features = [tf.contrib.layers.real_valued_column("x", dimension=2)]     # declare the feature. In this case 2D real number
# estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)  # build-in linear regression estimator. It is our model, is the front end to invoke training (fitting) and evaluation (inference)*[]:
#
# input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train, batch_size=2, num_epochs=1000)    # declare the input data
#
# estimator.fit(input_fn=input_fn, steps=1000)                            # learn the model
# print(estimator.evaluate(input_fn=input_fn))                            # evaluate loss function on the training data


# -------------USING TENSORFLOW LIBRARY FOR A CUSTOM MODEL-------------
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [2, 1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)      # extract the model params
    y = tf.matmul(features['x'], W) + b                            # declare the mode

    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))             # declare loss function

    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss= loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 2, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))


