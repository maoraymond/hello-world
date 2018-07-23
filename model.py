import tensorflow as tf
import pandas as pd
import numpy as np
import os
#from tensorflow.keras.utils import to_categorical
import datetime
import time
#from pandas._libs.lib import row_bool_subset
#from loadData import DataLoaders

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#[input_data, labels, test] = DataLoaders.get_input_and_labels(DataLoaders,13,6,100000)

# path = os.getcwd()
path = "/home/u12784/bio/BioProject/TF13"


def oneHot(string):
    trantab = str.maketrans('ACGT','0123')
    string = string+'ACGT'
    data = list(string.translate(trantab))
    return tf.keras.utils.to_categorical(data)[0:-4]   # .transpose()


def read_selex_csv_to_array(experiment_name, selex_num, rows_to_read, train_data=True):
    #path = os.path.join(os.path.expanduser('~'), 'hello-world','train/' if train_data else 'test/')
    path = "/home/u12784/bio/BioProject/TF13"
    while(1):
        selex_name = path +'/'+ experiment_name + '_selex_' + str(selex_num) + '.txt'
        if os.path.isfile(selex_name):
            break
        else:
            selex_num = selex_num-1
    print(selex_num)
    print("I stating to count rows")
    #rows_to_read = sum(1 for line in open(selex_name))
    print("I finished to count rows")
    selex = pd.read_csv(selex_name, sep="\t", header=-1, names=['seq', 'int'], nrows=rows_to_read)
    data = map(oneHot, selex['seq'])
    return np.array(list(data))

def read_pbm_to_array(ex_name, train_data=True):
    #path = os.path.join(os.path.expanduser('~'), 'hello-world', 'train/' if train_data else 'test/')
    path = "/home/u12784/bio/BioProject/TF13"
    pbm_name = path +'/'+ ex_name + '_pbm.txt'
    pbm = pd.read_csv(pbm_name, header=-1, names=['seq', 'int'])
    data = map(oneHot, pbm['seq'])
    test = np.array(list(data))
    return test[:, :36, :]
#    return np.append(test[:, :36, :], np.zeros([len(test), 40 - 36, 4]), axis=1)

def get_input_and_labels(n, max_selex_idx, rows_number):
    ex_name = 'TF' + str(n)
    test = read_pbm_to_array(ex_name, train_data=True)
    selex_0 = read_selex_csv_to_array(ex_name, selex_num=0, rows_to_read=rows_number, train_data=True)
    selex_4 = read_selex_csv_to_array(ex_name, selex_num=max_selex_idx, rows_to_read=rows_number, train_data=True)

    # create input data, labels
    input_data = np.append(selex_0, selex_4, axis=0)
    #input_data = np.append(input_data, np.zeros([len(input_data), 40 - selex_0.shape[1], 4]), axis=1)
    # input_data = np.append(input_data, np.zeros([len(input_data), 4, 40-selex_0.shape[2]]), axis=2)
    #labels = np.concatenate((np.zeros(len(selex_0)), np.ones(len(selex_4))))
    labels = np.zeros([len(input_data), 10])
    for i in range(0, 9):
        labels[1 + i * 100:(i + 1) * 100, i] = 1
    labels[901:, 9] = 1
    perm = np.random.permutation(labels.shape[0])
    return input_data[perm, :, :], labels[perm], test



# define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([6, 4, channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="VALID")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="SAME")


# fully connected layer
def fc_layer(input, channels_in, channels_out, name="FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act


rows_number=1000
[input_data, labels, test] = get_input_and_labels(13,6,rows_number)

tf.reset_default_graph()
sess = tf.Session()

# Set placeholders and reshape the data:
x = tf.placeholder(tf.float32, shape=[2*rows_number, input_data.shape[1], 4], name="input")
y = tf.placeholder(tf.float32, shape=[2*rows_number, 10], name="labels")
x_seq = tf.reshape(x, [-1, input_data.shape[1], 4, 1])
tf.summary.image("input", x_seq, 3)

# Create the network

# Convulotional layer
conv2 = conv_layer(x_seq, 1, 64, "conv1")
#conv2 = conv_layer(conv1, 32, 64, "conv2")
sizelist=conv2.get_shape().as_list()
flat_len_c2d = sizelist[1]*sizelist[2]*sizelist[3]
flattened = tf.reshape(conv2, [-1, flat_len_c2d])

# fully connected layer
fc1 = fc_layer(flattened, flat_len_c2d, 128, "fc1")
logits = fc_layer(fc1, 128, 10, "fc2")

# cross entropy - as loss function
with tf.name_scope("cross_entrupy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    tf.summary.scalar("cross_entropy", cross_entropy)

# Use Adam optimizer to train the network
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# compute the accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# combine all summaries:
merged_summary = tf.summary.merge_all()

# initialize all the variables
sess.run(tf.global_variables_initializer())

# tensorboard
writer = tf.summary.FileWriter(path + '/graph2')
writer.add_graph(sess.graph)

batch = tf.train.batch([x_seq, y], batch_size=100)
"""
# Train for 2001 step
for i in range(2001):
    batch = tf.train.batch([],100)
    if i % 5 == 0:
        [train_accuracy, s] = sess.run([accuracy, merged_summary], feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, i)

    # Occasionally report accuracy
    if i % 500 == 0:
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # Run the training step
"""
sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

"""
from sklearn.metrics import average_precision_score

predict = model.predict(np.array(test))
true = [int(x) for x in np.append(np.ones(100), np.zeros(len(test) - 100), axis=0)]
print(average_precision_score(true, predict))

"""
