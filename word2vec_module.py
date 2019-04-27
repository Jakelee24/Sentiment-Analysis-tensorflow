from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import urllib
import urllib.request
import tarfile
import os
import random
import string
import pickle
from Helper import Helper

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# training_data_pth = "JakeDrive/training-data-large.txt"
# for small is 124
# for large is 65466
# vocabulary_size = 65466
class Data_Processer:
    def __init__(self, training_data, vocabulary_size):
        basedir = os.getcwd()
        self.__training_data_pth = training_data
        self.__vocabulary_size = vocabulary_size
        # Take all the reviews associated with the training set
        # both pos and neg reviews
        # for each of these files
        # open it and read the contents into another file "all_data.txt"
        self.__target = os.path.join(basedir, 'all_data.txt')
        self.__CreateWordsFile()
        
    def __CreateWordsFile(self):
        with open(os.path.join(self.__target), "w") as fo:
            training_set = self.__training_data_pth
            train_df = pd.read_csv(training_set, sep= '\t', names=["Label", "Text"])
            train_text = train_df["Text"]
            for i in range(len(train_text)):
                fo.write(train_text[i])
                fo.write('\n')

    def __get_words_text(self,basedir):
        data_set = "all_data.txt"
        data_df = pd.read_csv(data_set,sep='\t', names=["Text"])
        data = data_df["Text"]
        li = list()
        for i in range(len(data)):
            temp = data[i].split(',')
            for j in range(len(temp)):
                li.append(temp[j])
        return li
    
    def build_dataset(self, basedir):
        words = self.__get_words_text(basedir)
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.__vocabulary_size))
        dictionary = dict()
        dictionary["<PAD>"] = len(dictionary)
        for word, _ in count:
            # dictionary contains words as keys, values are the occurrence rank
            # ie most common word has value 1
            # second value two
            # etc
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count = unk_count + 1
            # the index of the word (unique token) is equal to its occurrence rank
            # data contains all the words' unique tokens in order, as they appear
            # in the text file with all the reviews
            data.append(index)
        count[0][1] = unk_count
        # reverse dict enables us to go from token for word (occurrence rank)
        # to the actual word
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        del words
        return data, count, dictionary, reverse_dictionary


    def generate_batch(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        data_index = 0
        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return batch, labels


    def batch_tester(self, data, reverse_dictionary):
        print('data:', [reverse_dictionary[di] for di in data[:8]])
        for num_skips, skip_window in [(2, 1), (4, 2)]:
            batch, labels = self.generate_batch(data, batch_size=8, num_skips=num_skips, skip_window=skip_window)
            print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
            print('    batch:', [reverse_dictionary[bi] for bi in batch])
            print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

    def plot_with_labels(self, low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            x = round(x,4)
            y = round(y,4)
            plt.scatter(x, y)
            plt.annotate(str(label),
                            xy=(x, y),
                            xytext=(5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom')

        plt.savefig(filename)

class Word2Vec:
    def __init__(self, batch_size, num_skips, skip_window, embedding_size, vocabulary_size ,ckpt_path):
        self.__batch_size = batch_size
        self.__embedding_size = embedding_size
        self.__vocabulary_size = vocabulary_size
        self.__num_skips = num_skips
        self.__skip_window = skip_window
        self.__ckpt_path = ckpt_path
        self.__graph = None
        self.__loss = None
        self.__optimizer = None
        self.__similarity = None
        self.__valid_size = None
        self.__valid_window = None
        self.__normalized_embeddings = None

    def create_plt(self, reverse_dictionary):
        final_embeddings = self.__normalized_embeddings.eval()
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #plot_only = 500
        plot_only = 100
        low_dim_embs = tsne.fit_transform(final_embeddings[1:plot_only+1, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        data_preprocess.plot_with_labels(low_dim_embs, labels)

        # from here, need to take the embedding parameters and then pass them to the next stage of the
        # system - the sentiment analyser
        # ie , save final_embeddings. This has been done in the actual operation.

    def __generate_batch(self, data, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        data_index = 0
        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        return batch, labels

    def train(self, data, reverse_dictionary):
        # This helps us terminate early if training started before.
        started_before = False
        with tf.Session(graph=self.__graph) as session:
            # want to save the overall state and the embeddings for later.
            # I think we can do this in one, but I haven't had time to test this yet.
            # TODO make this a bit more efficient, avoid having to save stuff twice.
            # NOTE - this part is very closely coupled with the lstm.py script, as it
            # reads the embeddings from the location specified here. Might be worth
            # relaxing this dependency and passing the save location as a variable param.
            ckpt = tf.train.get_checkpoint_state(self.__ckpt_path)
            saver = tf.train.Saver(tf.global_variables())
            saver_embed = tf.train.Saver({'embeddings': self.__embeddings})
            if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
                saver.restore(session, ckpt.model_checkpoint_path)
                print("done")
                started_before = True
            else:
                print("Creating model with fresh parameters.")
                tf.global_variables_initializer().run()
                print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.__generate_batch(data, self.__batch_size, 
                self.__num_skips, self.__skip_window)

                feed_dict = {self.__train_dataset: batch_data, self.__train_labels: batch_labels}
                _, l = session.run([self.__optimizer, self.__loss], feed_dict=feed_dict)
                average_loss += l

                if step >= 10000 and (average_loss / 2000) < 5 and started_before:
                    print('early finish as probably loaded from earlier')
                    break

                if step % steps_per_checkpoint == 0:
                    # save stuff
                    checkpoint_path = os.path.join(ckpt_path, "model_ckpt")
                    embed_path = os.path.join(ckpt_embed,"embeddings_ckpt")
                    saver.save(session, checkpoint_path, global_step=self.__global_step)
                    saver_embed.save(session, embed_path)
                    print(embed_path)
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = self.__similarity.eval()
                    for i in range(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            #if nearest[k]<len(reverse_dictionary)
                            close_word = reverse_dictionary[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)

    def optmize(self, valid_size, valid_window, num_sampled):
        self.__valid_size = valid_size
        self.__valid_window = valid_window
        valid_examples = np.array(random.sample(range(valid_window), valid_size))
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            # variable to track progress
            self.__global_step = tf.Variable(0, trainable=False)

            # Input data.
            self.__train_dataset = tf.placeholder(tf.int32, shape=[self.__batch_size])
            self.__train_labels = tf.placeholder(tf.int32, shape=[self.__batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            with tf.device('/cpu:0'):
                # Variables.
                self.__embeddings = tf.Variable(tf.random_uniform([self.__vocabulary_size, self.__embedding_size], -1.0, 1.0), 
                                        name = "embeddings")
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.__vocabulary_size, self.__embedding_size],
                                        stddev=1.0 / math.sqrt(self.__embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.__vocabulary_size]))

                # Model.
                # Look up embeddings for inputs.
                # note that the embeddings are Variable params that will
                # be optimised!
                embed = tf.nn.embedding_lookup(self.__embeddings, self.__train_dataset)
            # Compute the nce loss, using a sample of the negative labels each time.
            # tried using sampled_softmax_loss, but performance was worse, so decided
            # to use NCE loss instead. Might be worth some more testing, especially with
            # the hyperparameters (ie num_sampled), to see what gives the best performance.
            self.__loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, self.__train_labels,
                                embed, num_sampled, self.__vocabulary_size))

            # PART BELOW LIFTED FROM TF EXAMPLES
            # Optimizer.
            # Note: The optimizer will optimize the nce weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities 
            # that contribute to the tensor it is passed.
            # See docs on `tf.train.Optimizer.minimize()` for more details.
            self.__optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.__loss, global_step=self.__global_step)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.__embeddings), 1, keepdims =True))
            self.__normalized_embeddings = self.__embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                self.__normalized_embeddings, valid_dataset)
            self.__similarity = tf.matmul(valid_embeddings, tf.transpose(self.__normalized_embeddings))


# Module usage example
if __name__ == '__main__':

    batch_size = 128 # how many target/context words to get in each batch
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right - context size
    num_skips = 2  # How many times to reuse an input to generate a label
    vocabulary_size = 65466
    # TAKEN FROM TF WEBSITE EXAMPLE:
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. 
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_sampled = 64# Number of negative examples to sample.

    #num_steps = 50001  # steps to run for
    num_steps = 1001
    steps_per_checkpoint = 50 # save the params every 50 steps.
    data_preprocess = Data_Processer("training-data-large.txt", vocabulary_size)

    basedir = os.getcwd()
    data, count, dictionary, reverse_dictionary = data_preprocess.build_dataset(basedir)

    # save the dictionary to file - very important for Data Processor
    Helper.store_stuff(dictionary, "dictionary.pickle", reverse_dictionary, "reverse_dictionary.pickle")
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    data_preprocess.batch_tester(data, reverse_dictionary)
    print('three index', dictionary['X773579']) # X773579 is the sample data, it cant replace to any other word
    ckpt_path = os.path.join(basedir, 'checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt_embed = os.path.join(ckpt_path, "embeddings")
    if not os.path.exists(ckpt_embed):
        os.makedirs(ckpt_embed)
    
    # finish the data preprocessing, now let's do the word2vec computation
    word2vec = Word2Vec(batch_size, num_skips, skip_window, embedding_size, vocabulary_size, ckpt_path)

    word2vec.optmize(valid_size, valid_window, num_sampled)
    word2vec.train(data, reverse_dictionary)
