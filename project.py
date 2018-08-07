import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score
import sys

# find file number
inFile = sys.argv[1]
TF_NUM = inFile[2:inFile.find("_")]

# local variables
ROWS_NUM=60000
BATCH_SIZE = 200
BUFFER = 200
epsilon = 1e-3
path = "/home/u12784/bio/test"


def oneHot(string): # return oneHot encoding of a seq
    trantab = str.maketrans('ACGT','0123')
    string = string+'ACGT'
    data = list(string.translate(trantab))
    return tf.keras.utils.to_categorical(data)[0:-4]   # .transpose()


def read_selex_csv_to_array(experiment_name, selex_num, rows_to_read,path): # reads selex data

    # find highest num selex
    while(1):
        selex_name = path +'/'+ experiment_name + '_selex_' + str(selex_num) + '.txt'
        if os.path.isfile(selex_name):
            break
        else:
            selex_num = selex_num-1
    selex = pd.read_csv(selex_name, sep="\t", header=-1, names=['seq', 'int'], nrows=rows_to_read)

    return list(selex['seq'])


def read_pbm_to_array(ex_name,path):
    pbm_name = path +'/'+ ex_name + '_pbm_unsorted.txt'
    print ('1')
    print (pbm_name)
    pbm = pd.read_csv(pbm_name, header=-1, names=['seq', 'int'])
    data = map(oneHot, pbm['seq'])
    test = np.array(list(data))
    print ('1')
    return test[:, :36, :] # read only 36 letters (the rest are the same)

def read_pbm_to_array2(ex_name,path): #read as letter and not as oneHot
    pbm_name = path +'/'+ ex_name + '_pbm_unsorted.txt'
    pbm = pd.read_csv(pbm_name, header=-1, names=['seq', 'int'])
    return list(pbm['seq'])

def get_input_and_labels(n, max_selex_idx, rows_number): # read data and label it
    ex_name = 'TF' + str(n)
    # read data
    test = read_pbm_to_array(ex_name,path=path)
    selex_0 = read_selex_csv_to_array(ex_name, selex_num=0, rows_to_read=rows_number,path=path)
    selex_4 = read_selex_csv_to_array(ex_name, selex_num=max_selex_idx, rows_to_read=rows_number,path=path)
    # remove same data in selex 0 and selex 4 from selex 0
    selex_0 = list(set(selex_0)-set(selex_4))
    # map to oneHot
    selex_0 = np.asarray(list(map(oneHot, selex_0)))
    selex_4 = np.asarray(list(map(oneHot, selex_4)))
    # create input data, labels
    input_data = np.append(selex_0, selex_4, axis=0)
    if(input_data.shape[1] ==40):
        input_data = input_data[:,:36,:]
    labels = np.zeros([len(input_data),2])
    #label the selex data - Bind/not Bind
    batch_len = round(rows_number)
    for i in range(2):
        labels[i * batch_len:(i + 1) * batch_len, i] = 1
    # mix indexes
    perm = np.random.permutation(labels.shape[0])
    #Params
    seq_len = input_data.shape[1]
    test_data_len = test.shape[0]
    test_len = test.shape[1]
    # reorder test element from [aaaa,bbbb] to [aa,aa,bb,bb] while 4 is pbm_seq_len and 2 is NumSubSeqs
    NumSubSeqs = (test_len - seq_len) // 6 + 1 # Strides = 6
    emptytest = np.empty((0, seq_len, 4))
    for i in range(0, NumSubSeqs):
        left = i*6
        right = seq_len + i*6
        emptytest = np.append(emptytest, test[:, left:right, :], axis=0)
    if (36 - seq_len) < 0 :
        print (emptytest.shape)
        print (test.shape)
        test = np.append(emptytest,test,axis=0)
    else: # if selex bigger than 36
        test = np.append(emptytest, test[:, 36 - seq_len:, :], axis=0)

    return input_data[perm, :, :], labels[perm], test ,test_data_len, NumSubSeqs+1



# define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name): #tensorboard
        #sets weights and bias
        w = tf.Variable(tf.truncated_normal([12, 1, channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        #convulotion
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        #batch-normalization params
        batch_mean, batch_var = tf.nn.moments(conv, [0])
        scale = tf.Variable(tf.ones([channels_out]))
        #activate BN
        BN = tf.nn.batch_normalization(conv, batch_mean, batch_var, b, scale, epsilon)
        #act function
        act = tf.nn.relu(BN)
        #write to tensor board
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        # max-pooling saves oneHot form (4)
        return tf.nn.max_pool(act, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding="SAME")


# fully connected layer
def fc_layer(input, channels_in, channels_out, name="FC"):
    with tf.name_scope(name): #tensorboard
        #sets weights and bias
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        #multiply
        z = tf.matmul(input, w)
        #batch-normalization params
        batch_mean, batch_var = tf.nn.moments(z, [0])
        scale = tf.Variable(tf.ones([channels_out]))
        #activate BN
        BN = tf.nn.batch_normalization(z, batch_mean, batch_var, b, scale, epsilon)
        #act function
        act = tf.nn.relu(BN)

        return act


def logits_to_perc(labels, test_len,NumSubSeqs): #from output to percentage
    #choose only bind nueron
    a = tf.constant([0,1], shape=[2, 1],dtype=tf.float32)
    bind_perc = tf.matmul(labels, a)
    # reorder PBM : ['AB','EF','IJ','CD','GH','KL'] -> ['AB','CD','EF','GH','IJ','KL']
    ind = []
    for i in range(test_len):
        for j in range(NumSubSeqs):
            ind = np.append(ind,[i + j * test_len], axis=0)
    ind = ind.astype(int)
    bind_perc = tf.gather(bind_perc,ind)
    # max pooling to get best result
    bind_perc = tf.reshape(bind_perc, [1, test_len * NumSubSeqs, 1, 1])
    bind_perc= tf.nn.max_pool(bind_perc,[1,NumSubSeqs,1,1],[1,NumSubSeqs,1,1],padding="VALID")

    return bind_perc


# reset session and tensorboard
tf.reset_default_graph()
sess = tf.Session()

# get data
[input_data, labels, test, test_len, NumSubSeqs] = get_input_and_labels(TF_NUM,6,ROWS_NUM)
#train data
train_data = [input_data, labels]
#test label (unused) - just for formolarity
true_order = [int(x) for x in np.append(np.ones(100), np.zeros(len(test) - 100), axis=0)]
true_order = np.reshape(true_order,[len(true_order),1])
true_order = np.repeat(true_order,2,axis=1)
#test data
test_data = [test, true_order]

# Set placeholders and reshape the data:
x = tf.placeholder(tf.float32, shape=[None, input_data.shape[1] , 4], name="input")
y = tf.placeholder(tf.float32, shape=[None,2], name="labels")
x_seq = tf.reshape(x, [-1, input_data.shape[1] * 4, 1, 1]) # reshpae data into one dimention
batch_size = tf.placeholder(tf.int64)

# Set database:
dataset = tf.data.Dataset.from_tensor_slices((x_seq, y)).repeat().batch(batch_size)

# init data
iter = dataset.make_initializable_iterator()
seqs, seqs_val = iter.get_next()

# Create the network

# Convulotional layer
#1st layer
conv1 = conv_layer(seqs, 1, 32, "conv1")
# conv1 = tf.nn.dropout(conv1,0.5) #dropout
#2nd layer
conv2 = conv_layer(conv1, 32, 64, "conv2")
# conv2 = tf.nn.dropout(conv2,0.5) #dropout
# conv3 = conv_layer(conv2, 64, 112, "conv3")
#flatten data
sizelist=conv2.get_shape().as_list()
flat_len_c2d = sizelist[1]*sizelist[2]*sizelist[3]
flattened = tf.reshape(conv2, [-1, flat_len_c2d])

# fully connected layer
fc1 = fc_layer(flattened, flat_len_c2d, 128, "fc1")
# fc1 = tf.nn.dropout(fc1,0.5) #dropout
#output
logits = fc_layer(fc1, 128, 2, "fc2")



# cross entropy - as loss function
with tf.name_scope("cross_entrupy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=seqs_val))
    tf.summary.scalar("cross_entropy", cross_entropy) #tensorboard

# compute the accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(seqs_val, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy) #tensorboard

# Use Adam optimizer to train the network
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy - accuracy)


# combine all summaries:
merged_summary = tf.summary.merge_all()

# initialize all the variables
sess.run(tf.global_variables_initializer())

# write to tensorboard
writer = tf.summary.FileWriter(path + '/graph')
writer.add_graph(sess.graph)

# intilize
sess.run(iter.initializer, feed_dict={x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})


# Train for 1051 step - 7 epochs

for i in range(1051):
    sess.run(train_step, feed_dict={x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
    if i % 5 == 0:
        # check accuracy and loss while taining
        [train_accuracy, loss ,s] = sess.run([accuracy, cross_entropy, merged_summary], feed_dict={x: train_data[0], y: train_data[1] , batch_size: BATCH_SIZE})
        # write to tensorboard
        writer.add_summary(s, i)

    # Occasionally report accuracy
    # if i % 100 == 0:
    #      print("step %d, training accuracy %g loss is %g" % (i, train_accuracy, loss))

# test AUPR of PBM

# init session with test data
sess.run(iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})

with tf.name_scope("AUPR"):
    test_label = logits_to_perc(logits, test_len, NumSubSeqs) # get percentage
    # create label to test
    aupr_true = [int(x) for x in np.append(np.ones(100), np.zeros(test_len - 100), axis=0)]
    aupr_true = np.reshape(aupr_true, [test_len])
    test_label = tf.reshape(test_label,[test_len])
    aupr_true = tf.convert_to_tensor(aupr_true)

    #check AUPR as asked
    with sess.as_default():
        AUPR_test = average_precision_score(aupr_true.eval(),test_label.eval())
        print('Test AUPR: {:4f}'.format(AUPR_test))
        # read PBM again
        PBM_list = read_pbm_to_array2('TF' + str(TF_NUM),path=path)
        # # sort test labels
        Plabels = test_label.eval()
        Plabels = np.asarray(Plabels)
        argsort = np.argsort(Plabels, axis=-1, kind='quicksort', order=None)
        #print to file
        for i in reversed(argsort):
            print("%s" % PBM_list[i], file=open(path +'/1'+ '/PBM' + str(TF_NUM) + '_sorted.txt', "a"))

#print AUPR to file
print("%g," % AUPR_test, file=open(path + "/AUPR_results_with_3conv.txt", "a"))


