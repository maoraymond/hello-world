import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import average_precision_score

inFile = sys.argv[1]
TF_NUM = inFile[2:inFile.find("_")]

# local variables
#TF_NUM = 13
ROWS_NUM=10000
EPOCHS = 5
BATCH_SIZE = 200
BUFFER = 200
epsilon = 1e-3
TB_path = "/home/u12784/bio/maor/train"
#TB_path = os.getcwd()


def oneHot(string):
    trantab = str.maketrans('ACGT','0123')
    string = string+'ACGT'
    data = list(string.translate(trantab))
    return tf.keras.utils.to_categorical(data)[0:-4]   # .transpose()


def read_selex_csv_to_array(experiment_name, selex_num, rows_to_read, train_data=True):
    path = "/home/u12784/bio/train"
    #path = os.getcwd()
    while(1):
        selex_name = path +'/'+ experiment_name + '_selex_' + str(selex_num) + '.txt'
        if os.path.isfile(selex_name):
            break
        else:
            selex_num = selex_num-1
    print("selex name is %s"%(selex_name))
    selex = pd.read_csv(selex_name, sep="\t", header=-1, names=['seq', 'int'], nrows=rows_to_read)
    data = map(oneHot, selex['seq'])
    return list(data)

def read_pbm_to_array(ex_name, train_data=True):
    # path = os.path.join(os.path.expanduser('~'), 'hello-world', 'train/' if train_data else 'test/')
    path = "/home/u12784/bio/train"
    #path = os.getcwd()
    pbm_name = path +'/'+ ex_name + '_pbm.txt'
    print(pbm_name)
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

    selex_0 = list(set(selex_0)-set(selex_4))
    selex_0 = np.asarray(selex_0)
    selex_4 = np.asarray(selex_4)  
    # create input data, labels
    input_data = np.append(selex_0, selex_4, axis=0)
    labels = np.zeros([len(input_data),2])
    batch_len = round(rows_number)
    for i in range(2):
        labels[i * batch_len:(i + 1) * batch_len, i] = 1
    perm = np.random.permutation(labels.shape[0])
    seq_len = input_data.shape[1]
    test_data_len = test.shape[0]
    test_len = test.shape[1]
    NumSubSeqs = (test_len - seq_len) // 6 + 1
    # reorder test element from [aaaa,bbbb] to [aa,aa,bb,bb] while 4 is pbm_seq_len and 2 is NumSubSeqs
    emptytest = np.empty((0, seq_len, 4))
    for i in range(0, NumSubSeqs):
        left = i*6
        right = seq_len + i*6
        emptytest = np.append(emptytest, test[:, left:right, :], axis=0)
    if (36 - seq_len) < 0 :
        test = np.append(emptytest,test,axis=0)
    else:
        test = np.append(emptytest, test[:, 36 - seq_len:, :], axis=0)
    return input_data[perm, :, :], labels[perm], test ,test_data_len, NumSubSeqs+1



# define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([12, 1, channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        batch_mean, batch_var = tf.nn.moments(conv, [0])
        scale = tf.Variable(tf.ones([channels_out]))
        BN = tf.nn.batch_normalization(conv, batch_mean, batch_var, b, scale, epsilon)
        act = tf.nn.relu(BN)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding="SAME")


# fully connected layer
def fc_layer(input, channels_in, channels_out, name="FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        z = tf.matmul(input, w)
        batch_mean, batch_var = tf.nn.moments(z, [0])
        scale = tf.Variable(tf.ones([channels_out]))
        BN = tf.nn.batch_normalization(z, batch_mean, batch_var, b, scale, epsilon)
        act = tf.nn.relu(BN)
        return act


def logits_to_perc(labels, test_len,NumSubSeqs):
    a = tf.constant([0,1], shape=[2, 1],dtype=tf.float32)
    bind_perc = tf.matmul(labels, a)
    ind = []
    for i in range(test_len):
        for j in range(NumSubSeqs):
            ind = np.append(ind,[i + j * test_len], axis=0)
    ind = ind.astype(int)
    bind_perc = tf.gather(bind_perc,ind)
    bind_perc = tf.reshape(bind_perc, [1, test_len * NumSubSeqs, 1, 1])
    bind_perc= tf.nn.max_pool(bind_perc,[1,NumSubSeqs,1,1],[1,NumSubSeqs,1,1],padding="VALID")
    return bind_perc

# reset session and tensorboard
tf.reset_default_graph()
sess = tf.Session()

# get data
[input_data, labels, test, test_len, NumSubSeqs] = get_input_and_labels(TF_NUM,6,ROWS_NUM)
train_data = [input_data, labels]
true_order = [int(x) for x in np.append(np.ones(100), np.zeros(len(test) - 100), axis=0)]
true_order = np.reshape(true_order,[len(true_order),1])
true_order = np.repeat(true_order,2,axis=1)
test_data = [test, true_order]

# Set placeholders and reshape the data:
x = tf.placeholder(tf.float32, shape=[None, input_data.shape[1] , 4], name="input")
y = tf.placeholder(tf.float32, shape=[None,2], name="labels")
x_seq = tf.reshape(x, [-1, input_data.shape[1] * 4, 1, 1])
batch_size = tf.placeholder(tf.int64)
#tf.summary.image("input", x_seq, 3)

# Set database:
dataset = tf.data.Dataset.from_tensor_slices((x_seq, y)).repeat().batch(batch_size)
#dataset = dataset.shuffle(buffer_size=BUFFER)

# init data
iter = dataset.make_initializable_iterator()
seqs, seqs_val = iter.get_next()

# Create the network

# Convulotional layer
conv1 = conv_layer(seqs, 1, 32, "conv1")
# conv1 = tf.nn.dropout(conv1,0.75)
conv2 = conv_layer(conv1, 32, 64, "conv2")
# conv2 = tf.nn.dropout(conv2,0.75)
sizelist=conv2.get_shape().as_list()
flat_len_c2d = sizelist[1]*sizelist[2]*sizelist[3]
flattened = tf.reshape(conv2, [-1, flat_len_c2d])

# fully connected layer
fc1 = fc_layer(flattened, flat_len_c2d, 128, "fc1")
# fc1 = tf.nn.dropout(fc1,0.75)
logits = fc_layer(fc1, 128, 2, "fc2")



# cross entropy - as loss function
with tf.name_scope("cross_entrupy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=seqs_val))
    tf.summary.scalar("cross_entropy", cross_entropy)

# compute the accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(seqs_val, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Use Adam optimizer to train the network
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy - accuracy)


# combine all summaries:
merged_summary = tf.summary.merge_all()

# initialize all the variables
sess.run(tf.global_variables_initializer())

# tensorboard
writer = tf.summary.FileWriter(TB_path + '/graph_6tep')
writer.add_graph(sess.graph)


sess.run(iter.initializer, feed_dict={x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())

# Train for 20001 step

for i in range(20001):
    sess.run(train_step, feed_dict={x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
    if i % 5 == 0:
        # check accuracy and loss while taining
        [train_accuracy, loss ,s] = sess.run([accuracy, cross_entropy, merged_summary], feed_dict={x: train_data[0], y: train_data[1] , batch_size: BATCH_SIZE})
        # check test (AUPR of PBM)
        writer.add_summary(s, i)

    # Occasionally report accuracy
    if i % 100 == 0:
        print("step %d, training accuracy %g loss is %g" % (i, train_accuracy, loss))


# test AUPR of PBM
sess.run(iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})

with tf.name_scope("AUPR"):
    test_label = logits_to_perc(logits, test_len, NumSubSeqs)
    aupr_true = [int(x) for x in np.append(np.ones(100), np.zeros(test_len - 100), axis=0)]
    aupr_true = np.reshape(aupr_true, [len(aupr_true)])
    test_label = tf.reshape(test_label,[len(aupr_true)])
    aupr_true = tf.convert_to_tensor(aupr_true)
    with sess.as_default():
        AUPR_test = average_precision_score(aupr_true.eval(),test_label.eval())
        print('Test AUPR: {:4f}'.format(AUPR_test))

print("****%g" % AUPR_test, file=open(TB_path + "/AUPR_file.txt", "a"))



