import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

milk = pd.read_csv('monthly-milk-production.csv',index_col='Month')
milk.index = pd.to_datetime(milk.index)

train_set = milk.head(156)
test_set = milk.tail(12)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

# Just one feature, the time series
num_inputs = 1
# Num of steps in each batch
num_time_steps = 12
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1

learning_rate = 0.03 
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 1000
# Size of the batch of data
batch_size = 1

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# Also play around with GRUCell
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs) 

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

def next_batch(training_data,batch_size,steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps) 
    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 

sess = tf.InteractiveSession()
sess.run(init)

for iteration in range(num_train_iterations):
    X_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)
    sess.run(train, feed_dict={X: X_batch, y: y_batch})
    if iteration % 100 == 0:
        mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
        print(iteration, "MSE:", mse)
        
# Save Model for Later
model_dir = './time_series_model/'
saver.save(sess, model_dir)

serialized_tf_example = tf.placeholder(tf.string, name='x')
feature_configs = {
    'x': tf.FixedLenFeature(
        shape=[num_time_steps, num_inputs], dtype=tf.float32),
    }
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
input_x = tf_example['x']

SavedModel_folder = "SavedModel1"
builder = tf.saved_model.builder.SavedModelBuilder(SavedModel_folder)

predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(input_x)
predict_tensor_outputs_info = tf.saved_model.utils.build_tensor_info(outputs)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'inputs': predict_tensor_inputs_info},
        outputs={'outputs': predict_tensor_outputs_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
)

# save the model
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_images': prediction_signature
    },
    legacy_init_op=legacy_init_op)

builder.save()
