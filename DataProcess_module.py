import string
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from Helper import Helper

class PreProcess:

    def __init__(self, data_file_pth, dict, rever_dict):
        if not os.path.exists("processed"):
            os.makedirs("processed/")
            print("No processed data found - generating")
        else:
            print("Some data may exist!!")
        self.__loadDictionary(dict, rever_dict)
        self.__data_file_pth = data_file_pth

    def __loadDictionary(self, dict, rever_dict):
        if not os.path.exists(dict) or not os.path.exists(rever_dict):
            print("Can't find dictionary, please create dictionary first!")
            return
        self.__word_ids, self.__rev_word_ids = Helper.get_stuff(dict, rever_dict)

    def __cleanup_test(self,text):
        li = text.split(',')
        return li

    def __save_data(self,npArray, path):
        start = os.getcwd()
        path = os.path.join(start, path)
        np.save(path, npArray)

    def __process_review(self ,review_data, word_ids, max_seq_length, data_type, data_id):
        data = np.array([i for i in range(max_seq_length)])
        word_indices = []
        
        processed_dir = os.path.join("processed", data_type)
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        new_name = data_id + ".npy"
        processed_path = os.path.join(processed_dir, new_name)
        if os.path.exists(processed_path):
            print(processed_path)
            return processed_path
        
        word_list = self.__cleanup_test(review_data)
        for word in word_list:
            if word in word_ids:
                token = word_ids[word]
        else:
            token = word_ids["UNK"]
        word_indices.append(token)
        
        if len(word_indices) < max_seq_length:
            padding = word_ids["<PAD>"]
            assert(padding == 0)
            word_indices = word_indices + [padding for i in range(max_seq_length - len(word_indices))]
        else:
            word_indices = word_indices[0:max_seq_length]
        data = np.vstack((data, word_indices))
        # data is set up to have the right size, but the first row is just full of dummy data.
        # this slicing procedure extracts JUST the word indices.
        data = data[1::]
        self.__save_data(data, processed_path)
        return processed_path

    def __seq_data(self, data_unsep):
        data_seq = list()
        for data in data_unsep:
            data_seq.append(data)
        return data_seq

    def create_processed_files(self, data_label ,data_texts, max_seq_length, datatype):
        internal_list = []
        li_test = []
        li_train = []

        for i in range(len(data_texts)):
            output_file = self.__process_review(data_texts[i], self.__word_ids, max_seq_length, datatype, str(i))
            if output_file is not None:
                internal_list.append((output_file, data_label[i]))
        
        print (internal_list[:10])
        # very important to store these so that LSTM can use them.
        if datatype =="test":
            li_test.extend(internal_list)
            Helper.store_stuff(li_test, "li_test.pickle")
        else:
            li_train.extend(internal_list)
            Helper.store_stuff(li_train, "li_train.pickle")

    def split_data(self, percent):
        data_df = pd.read_csv(self.__data_file_pth, sep='\t', names=["Label", "Text"])
        data_text = data_df["Text"]
        data_label = data_df["Label"]

        # example: set percent to 0.2 it will be 80% for training 20% for test
        train_datas, test_datas, train_labels, test_labels = train_test_split(data_text, 
        data_label, test_size=percent, random_state=42)

        return self.__seq_data(train_datas), self.__seq_data(test_datas),\
        self.__seq_data(train_labels), self.__seq_data(test_labels)
    


# Module usage example
if __name__ == '__main__':
    # step 1: init preprocess first
    preprocess = PreProcess("training-data-large.txt", "dictionary.pickle", "reverse_dictionary.pickle")

    # step 2: split the train data to 80% of training 20% of testing by this case
    data_trains, data_tests, label_trains, label_tests = preprocess.split_data(0.2)

    # step 3: create the train and test file for the LSTM
    max_seq_length = 120
    preprocess.create_processed_files(label_trains, data_trains, max_seq_length, "train")
    preprocess.create_processed_files(label_tests, data_tests, max_seq_length, "test")
