import tensorflow as tf
import pickle

from tensorflow.nn import rnn_cell
from tensorflow.contrib.rnn.python.ops import rnn

from tensorflow.python.platform import gfile

import numpy as np
import os
from Helper import Helper

class DataFeeder:
    # This class is made to hold the data that should be input into the system for processing
    # init function converts stored numpy arrays into an in-memory thing that can be
    # passed in batches to the model
    def __init__(self, trainpath=None, testpath=None, batch_size=None):
        # this init function takes the stored numpy arrays of word indices and converts
        # them into a suitable format for passing to the model

        # lists with data/labels
        self.train_data = []
        self.test_data = []

        # indexes for the data, needed for fetching a training batch
        self.train_data_index = 0
        self.test_data_index = 0

        # first load the filepaths and labels into memory
        # li_train and li_test are lists of tuples
        # li_train[i][0] is the path to the i^th data example
        # actual file stored at this location is a numpy array
        # li_train[i][1] is the label associated with that example

        # to do - pass this as an arg?
        if trainpath is None and testpath is None:
            li_train, li_test = Helper.get_stuff("li_train.pickle", "li_test.pickle")

        # now store the test/train data in the right arrays
        self._init_data(li_train, test=False)
        np.random.shuffle(self.train_data)
        self._init_data(li_test, test=True)
        np.random.shuffle(self.test_data)

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 200

        # arrays loaded with the right data - all done.

    def get_batch(self, test=False):
        # returns a batch of the training/test data, as requested
        # batch is of size self.batch_size
        # also returns the sequence length - how long is the review?
        # (in terms of num of words)
        # This is now done at batch fetch time. Ideally this is done
        # as part of the tensorflow computation graph, but using
        # zero padding is proving challenging...
        data_batch = []
        label_batch = []
        seq_length = []
        temp_index = 0
        if test:
            li = self.test_data
            temp_index = self.test_data_index % len(self.test_data)
            self.test_data_index = (self.test_data_index +self.batch_size) % len(self.test_data)
        else:
            li = self.train_data
            temp_index = self.train_data_index % len(self.train_data)
            self.train_data_index = (self.train_data_index + self.batch_size) % len(self.train_data)

        for _ in range(self.batch_size):
            np_data = li[temp_index][0]
            # look for zero elements - these are PADDING elements!
            # these are used to determine the length of the review.
            # TODO find out what's happening with this- something bizarre is going on
            # and makes indices 2D even when np_data is a single dimension?
            # Hack below with swallowing exception works, but is not a permanent fix
            # Solved - this should really be if len(indices.shape) > 1.
            # Haven't tested this fully, so remains a todo.
            indices = np.where(np_data == 0)
            if len(indices) > 0:
                try:
                    seq_len = indices[0][0]
                except IndexError:
                    seq_len = len(np_data)
            else:
                seq_len = len(np_data)
            # we've found how many elements are zero - append this to the return seq_length
            seq_length.append(seq_len)
            data_batch.append(np_data)
            label_batch.append(li[temp_index][1])
            if test:
                temp_index = ((temp_index + 1) % len(self.test_data))
            else:
                temp_index = ((temp_index + 1) % len(self.train_data))
        return data_batch, label_batch, seq_length

    def _init_data(self, li, test=False):
        temp = []
        count = 0
        for elem in li:
            data = np.load(elem[0])[0]
            if elem[1] == 1:
                # positive example
                label = np.array([1, 0])
            else:
                # negative example
                label = np.array([0, 1])
            # data contains the word indices as a np array
            # label contains the associated class label as a np array
            item = (data, label)
            temp.append(item)
        if test:
            self.test_data.extend(temp)
        else:
            self.train_data.extend(temp)
        del temp


class LSTM:
    def __init__(self, batch_size, basedir, steps_per_checkpoint):
        self.__batch_size = batch_size
        self.__feeder = DataFeeder(batch_size=batch_size)
        self.__steps_per_checkpoint = steps_per_checkpoint
    
    def __unpack_sequence(self, tensor):
        return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))

    def build_model(self,max_length, vocabulary_size, embedding_size, 
    num_hidden, num_layers,num_classes, learn_rate):
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            # to track progress
            self.__global_step = tf.Variable(0, trainable=False)

            # input parameters
            self.__dropout = tf.placeholder(tf.float32)
            self.__data = tf.placeholder(tf.int32, shape=[self.__batch_size, max_length], name="data")
            self.__labels = tf.placeholder(tf.float32, shape=[self.__batch_size, num_classes], name="labels")
            self.__seq_length = tf.placeholder(tf.int32, shape=[self.__batch_size], name="seqlength")

            # LSTM definition
            network = rnn_cell.LSTMCell(num_hidden, embedding_size)
            network = rnn_cell.DropoutWrapper(network, output_keep_prob=self.__dropout)
            network = rnn_cell.MultiRNNCell([network] * num_layers)

            # loaded value from word2vec
            self.__embeddings_lstm = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                                        name="embeddings_lstm")

            # get the word vectors learned previously
            embed = tf.nn.embedding_lookup(self.__embeddings_lstm, self.__data)
            #try:
            outputs, states = tf.contrib.rnn.static_rnn(network, self.__unpack_sequence(embed), dtype=tf.float32, sequence_length=self.__seq_length)
            #except AttributeError:
                #outputs, states = rnn(network, unpack_sequence(embed), dtype=tf.float32, sequence_length=seq_length)

            # Compute an average of all the outputs
            # FOR VARIABLE SEQUENCE LENGTHS 
            # place the entire sequence into one big tensor using tf_pack.
            packed_op = tf.stack(outputs)
            # reduce sum the 0th dimension, which is the number of timesteps.
            summed_op = tf.reduce_sum(packed_op, reduction_indices=0)
            # Then, divide by the seq_length input - this is an np array of size 
            # batch_size that stores the length of each sequence. 
            # With any luck, this gives the output results. 
            averaged_op = tf.div(summed_op, tf.cast(self.__seq_length, tf.float32))


            # output classifier
            # TODO perhaps put this within a context manager
            softmax_weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
            softmax_bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            temp = tf.matmul(averaged_op, softmax_weight) + softmax_bias
            prediction = tf.nn.softmax(temp)
            predict_output = tf.argmax(prediction, 1)
            tf.summary.histogram("prediction", prediction)
            self.__prin = tf.Print(prediction, [prediction], message="pred is ")
            # standard cross entropy loss
            self.__loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=temp, labels=self.__labels))
            tf.summary.scalar("loss-xentropy", self.__loss)

            self.__optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.__loss, global_step=self.__global_step)

            # examine performance
            with tf.variable_scope("accuracy"): 
                correct_predictions = tf.equal(tf.argmax(prediction, 1),tf.argmax(self.__labels, 1))
                self.__accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
            tf.summary.scalar("accuracy", self.__accuracy)


            self.__merged = tf.summary.merge_all()
            self.__train_writer = tf.summary.FileWriter(os.getcwd() + '/train', self.__graph)
            self.__test_writer = tf.summary.FileWriter(os.getcwd() + '/test')
    

    def train(self, old_ckpt_path, ckpt_path, num_train_steps):
        with tf.Session(graph=self.__graph) as session:
            # this is ugly - only for the embeddings, which were trained before.
            # TODO make this nicer!
            ckpt1 = tf.train.get_checkpoint_state(old_ckpt_path)
            ckpt2 = tf.train.get_checkpoint_state(ckpt_path)
            saver_lstm_embed = tf.train.Saver({"embeddings": self.__embeddings_lstm})
            saver = tf.train.Saver(tf.all_variables())

            started_before = False
            if ckpt2 and gfile.Exists(ckpt2.model_checkpoint_path):
                print("Reading model parameters from {0}".format(ckpt2.model_checkpoint_path))
                saver.restore(session, ckpt2.model_checkpoint_path)
                print("done")
                started_before = True
            else:
                print("Creating model with fresh parameters.")
                tf.initialize_all_variables().run()
                print('Initialized')

            if ckpt1 and gfile.Exists(ckpt1.model_checkpoint_path) and not started_before:
                print("Reading embeddings from {0}".format(ckpt1.model_checkpoint_path))
                embed_path = os.path.join(old_ckpt_path, "embeddings")
                saver_lstm_embed.restore(session, embed_path)
                print("done")

            average_loss = 0
            average_acc = 0
            for step in range(num_train_steps):
                batch_data, batch_labels, seq_len = self.__feeder.get_batch(test=False)
                feed_dict = {self.__data: batch_data, self.__labels: batch_labels, self.__dropout: 0.5,
                            self.__seq_length: seq_len}
                _, l, acc, summ = session.run([self.__optimizer, self.__loss, self.__accuracy, self.__merged], feed_dict=feed_dict)
                average_loss += l
                average_acc += acc
                if step % self.__steps_per_checkpoint == 0:
                    # save stuff
                    checkpoint_path = os.path.join(ckpt_path, "lstm_checkpoints")
                    saver.save(session, checkpoint_path, global_step=self.__global_step)
                    self.__train_writer.add_summary(summ,step)

                if step % 200 == 0:
                    if step > 0:
                        average_loss = average_loss / 200
                        average_acc = average_acc / 200
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    print('Average accuracy at step %d: %f' % (step, average_acc))
                    average_loss = 0
                    average_acc = 0

            num_test_steps = int(len(self.__feeder.test_data) / self.__batch_size)
            test_loss = 0
            test_accuracy = 0
            for _ in range(num_test_steps):
                batch_data, batch_labels, seq_len = self.__feeder.get_batch(test=True)
                feed_dict = {self.__data: batch_data, self.__labels: batch_labels, self.__dropout: 1,
                            self.__seq_length: seq_len}
                l, acc = session.run([self.__loss, self.__accuracy], feed_dict=feed_dict)
                test_accuracy += acc
                test_loss += l
            print('Test acc is  %f' % (test_accuracy/num_test_steps))
            print('Test loss is  %f' % (test_loss / num_test_steps))

# Module usage example
if __name__ == '__main__':

    # step 1: init the LSTM module
    batch_size = 200
    basedir = os.getcwd()
    steps_per_checkpoint = 100
    lstm = LSTM(batch_size, basedir, steps_per_checkpoint)

    # step 2: build LSTM model
    max_length = 120
    vocabulary_size = 65466
    embedding_size = 128
    num_hidden = 200
    num_layers = 1
    num_classes = 2
    learning_rate = 0.005
    lstm.build_model(max_length, vocabulary_size, embedding_size, 
    num_hidden, num_layers, num_classes, learning_rate)

    # step 3: start traning
    old_ckpt_path = os.path.join(basedir, "checkpoints", "embeddings")
    ckpt_path = os.path.join(basedir, 'lstm_checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    num_steps =1001
    lstm.train(old_ckpt_path, ckpt_path, num_steps)