
# coding: utf-8

# In[1]:


from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD,Adam
from keras.callbacks import CSVLogger
import json
import keras
import numpy as np
from keras.utils.np_utils import to_categorical
import keras.backend as K
import tensorflow as tf


# In[2]:


from tensorflow.python.client import device_lib


# In[3]:


print(device_lib.list_local_devices())


# In[4]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[5]:


docs = []
labels = []


# In[6]:


num_classes = 102


# In[7]:


def load_data(path1):
    temp_docs = []
    labels = []
    with open(path1) as json_file:
        data = json.load(json_file)
        
    for x in data:
        temp_docs.append(x['body'])
#         labels.append(keras.utils.to_categorical(np.array(list(x['middle_label'])),num_classes=102))
        labels.append((np.array(list(x['middle_label']))))
    len1 = len(temp_docs)
    print('number of human rights docs is: '+str(len1))
    return temp_docs,labels


# In[8]:


docs,labels = load_data('/home/tigermlt/CS341/github_repo/CS341/data/data_with_middle_layer_and_label.json')


# In[9]:


# docs_train = docs[:45000]
# labels_train = labels[:45000]
# docs_val = docs[45000:50000]
# labels_val = labels[45000:50000]
# docs_test = docs[50000:]
# labels_test = labels[50000:]
# docs = docs[:100]
# labels = labels[:100]


# In[10]:


print(docs[0])


# In[11]:


import random


# In[12]:


# random shuffle the data
c = list(zip(docs, labels))


# In[13]:


random.shuffle(c)


# In[14]:


docs, labels = zip(*c)


# In[15]:


docs = list(docs)
labels = list(labels)


# In[16]:


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)


# In[17]:


# compute max document length
max_val = -1
for i in range(len(docs)):
    temp_val = len(docs[i])
    if temp_val>max_val:
        max_val = temp_val
print(max_val)


# In[18]:


# pad documents to a max length, compute by calculating the maximum document length
max_length = 15260
# max_length = 5000
# max_length = 1068
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# In[19]:


print(len(padded_docs))


# In[20]:


# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[21]:


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[22]:


# # prepare validation data

# # prepare tokenizer
# t_val = Tokenizer()
# t_val.fit_on_texts(docs_val)
# # integer encode the documents
# encoded_docs_val = t_val.texts_to_sequences(docs_val)

# padded_docs_val = pad_sequences(encoded_docs_val, maxlen=max_length, padding='post')


# In[23]:


def build_model():
    # define model
    model = Sequential()
    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(LSTM(128,dropout = 0.8,implementation=2))
#     model.add(LSTM(128,dropout = 0.8,))
#     model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model


# In[24]:


model = build_model()


# In[25]:


# def euclidean_distance_loss(y_true, y_pred):
#     y_log = K.log(y_pred)
#     y_log_2 = K.log(1-y_pred)
    
#     part1 = -y_true*y_log
#     part2 = -(1-y_true)*y_log_2
    
#     matrix = part1 + part2
#     mean_row = K.mean(matrix,axis=1)
    
#     loss = K.mean(mean_row,axis=0)

#     return loss


# In[26]:


def accuracy(y_true, y_pred):
    diff = K.abs(y_true-y_pred)
    correct_num = K.sum(tf.to_int32(K.greater(0.1,diff)))
    accuracy = correct_num/16/102
    return accuracy


# In[27]:


csv_logger=CSVLogger('middle_class_v2.csv',append=True,separator=';')


# In[28]:


# compile the model
model.compile(optimizer=Adam(lr=0.01), loss=['binary_crossentropy'], metrics=['acc'])
# summarize the model
print(model.summary())


# In[ ]:


print(padded_docs.shape)


# In[ ]:


with tf.device('/device:GPU:0'):
    # fit the model
    model.fit(padded_docs, np.array(labels), epochs=10, validation_split = 0.10, batch_size = 256,callbacks=[csv_logger])

