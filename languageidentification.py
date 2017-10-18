
# coding: utf-8

# In[18]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import string
import numpy as np


# In[19]:


def sigmoid_array(x):
    # input: array
    # output: sigmoid applied to each value of input array
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # input: array
    # output: softmax of array
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))


def define_alphabet():
    # creates list of alphabet to use when parsing text data
    base_en = 'abcdefghijklmnopqrstuvwxyz'
    special_chars = ' !?¿¡' + string.punctuation + string.digits
    italian = 'àèéìíòóùú'
    french = 'àâæçéèêêîïôœùûüÿ'
    all_lang_chars = base_en + italian + french 
    small_chars = list(set(list(all_lang_chars)))
    small_chars.sort() 
    big_chars = list(set(list(all_lang_chars.upper())))
    big_chars.sort()
    small_chars += special_chars
    letters_string = ''
    letters = small_chars + big_chars
    for letter in letters:
        letters_string += letter
    return small_chars,big_chars,letters_string

alphabet, _, _ = define_alphabet()
print(alphabet, len(alphabet))


# In[20]:


# fits labelBinarizer obj to alphabet so it can make one-hot encoding
# c = unique characters between eng, fre, ital = 93
le = LabelBinarizer()
le.fit(alphabet)


# In[21]:


def forward(x, W1, bias1, W2, bias2):
    # input: 
        #(5c,1) input vector
        # (d,5c) weight matrix W1
        # (d,1) bias vector
        # (3,d) weight matrix W2
        # (3,1) bias vector
    # output:
        # (d, 1) hidden layer
        # (3, 1) predicted vector, y
    hidden_layer = sigmoid_array(np.dot(W1, x.T) + bias1)
    y = softmax(np.dot(W2, hidden_layer) + bias2)
    return hidden_layer, y



# In[22]:


def seq_chars(s, num_chars=5):
    # input: string, s
    # output: list of 5 sequential characters of string from beginning to end
    n = len(s)
    return [s[i:(i+num_chars)] for i in range(n-4)]

def binarize(seq_str):
    # input: sequence of characters
    # output: concatenated one hot encoding of each character
        # w/ dimension (5c, 1)
    nseq = len(seq_str)
    return np.array([le.transform(list(seq_str[i])) for i in range(nseq)]).reshape(nseq, input_dim)


# ## Backprop functions

# $\nabla_{y} L = y - \hat{y}$

# In[23]:


def grad_l_wrt_y(y_pred, y_true):
    return y_pred - y_true #(3,1)


# $\nabla_{b^{2}} L = \nabla_{y^{'}}L = \sum_{i} \frac{\delta{L}}{\delta{y_{i}}} y_{i}(\delta{ij} - y_{j}) = \sum_{i} (y_{i} - \hat{y_{i}}) y_{i}(\delta_{ij} - y_{j})$

# In[24]:


def grad_l_wrt_b2(y_pred, y_true):
    vec = []
    # for each yj
    for j, val in enumerate(list(y_pred)):
        counter = 0
        # go over all values in yi
        for i, val2 in enumerate(list(y_pred)):
            if i == j:
                counter += (y_pred[i] - y_test[i])*(y_pred[i])*(1-y_pred[j])
            else:
                counter += (y_pred[i] - y_test[i])*(y_pred[i])*(-y_pred[j])
        vec.append(counter)
    return np.array(vec)

     
  


# $\nabla_{w^{2}} L = \nabla_{y^{'}}L h^{T} = \nabla_{b^{2}}L h^{T}$

# In[25]:


def grad_l_wrt_w2(grad_b2, hidden_layer):
    # take grad wrt b^2 and mult by hidden layer
    return grad_b2.dot(hidden_layer.T) #(3,1)*(1,d) is (3,d)



# $\nabla_{h}L = W^{2T} \nabla_{y^{'}}L = W^{2T} \nabla_{b^{2}}L$

# In[26]:


def grad_l_wrt_h(grad_b2, W2):
    return np.dot(W2.T, grad_b2) #(d,3)*(3,1) is (d,1)


# $\frac{\delta{L}}{\delta{h^{'}_{i}}} = \frac{\delta{L}}{\delta{h_{i}}} h_{i}(1-h_{i})$ ie multiply each elt in $\nabla_{h}L$ by $h_{i}(1-h_{i})$

# In[36]:


def grad_l_wrt_h_tilde(grad_h, h_layer):
    # does element wise multiplication
    return grad_h * ((h_layer) * (1-h_layer)) # (d,1)(element mult)(d,1) is (d,1)

#     vec = []
#     for i in range(len(h_layer)):
# #         print("mult:", grad_h[i], (h_layer[i] * (1-h_layer[i])))
#         vec.append(grad_h[i] * (h_layer[i] * (1-h_layer[i])))

#     return np.array(vec)



# $\nabla_{w^{1}} L = (\nabla_{h^{'}}L)(x^{T})$

# In[37]:


def grad_l_wrt_w1(grad_h_tilde, input_x):
    return np.dot(grad_h_tilde, input_x) #(d,1)*(1,5c) is (d,5c)



# $\nabla_{b^{1}} L = (\nabla_{h^{'}}L)$

# In[29]:


def grad_l_wrt_b1(grad_h_tilde):
    return grad_h_tilde


# In[30]:



def backprop(y_pred, y_test, h_layer, input_x, W1, W2, bias1, bias2, eta=.1):   
    # input: ingredients for calculating gradients
    # output: W1, bias1, W2, bias2 after backprop
    
    grad_l_y = grad_l_wrt_y(y_pred, y_test)
#     print('grad_l_y shape is {}'.format(grad_l_y.shape))
#     print(grad_l_y)

    grad_l_b2 = grad_l_wrt_b2(y_pred, y_test)
#     print(grad_l_b2)
#     print('grad_l_b2 shape is {}'.format(grad_l_b2.shape))

    grad_l_w2 = grad_l_wrt_w2(grad_l_b2, h_layer)
#     print(grad_l_w2)
#     print('grad_l_w2 shape is {}'.format(grad_l_w2.shape))

    grad_l_h = grad_l_wrt_h(grad_l_b2, W2)
#     print(grad_l_h)
#     print('grad_l_h shape is {}'.format(grad_l_h.shape))

    grad_h_tilde = grad_l_wrt_h_tilde(grad_l_h, h_layer)
#     print(grad_h_tilde)
#     print('grad_h_tilde shape is {}'.format(grad_h_tilde.shape))

    grad_l_w1 = grad_l_wrt_w1(grad_h_tilde, input_x)
#     print(grad_l_w1)
#     print('grad_l_w1 shape is {}'.format(grad_l_w1.shape))

    grad_l_b1 = grad_l_wrt_b1(grad_h_tilde)
#     print(grad_l_b1)
#     print('grad_l_b1 shape is {}'.format(grad_l_b1.shape))


    W1 = W1 -  eta * grad_l_w1
    W2 = W2 - eta * grad_l_w2
    bias2 = bias2 - eta * grad_l_b2
    bias1 = bias1 - eta * grad_l_b1

    return W1, bias1, W2, bias2


# In[60]:


np.random.seed(1)
# one line of each language
filename = "languageIdentification.data/teeny_tiny_train.txt"
d = 100
eta = 0.1
input_dim = 455

W1 = np.random.uniform(size=(d, input_dim))
bias1 = np.random.uniform(size=[d,1])
W2 = np.random.uniform(size=(3,d))
bias2 = np.random.uniform(size=[3, 1])
eta = 0.1
accuracy_li = []
loss_li = []


# In[61]:


for i in range(300): # number of epochs where training data is 3 sentences
    with open(filename, 'r') as handle:
        num_chances = 0 # number of lines in text file
        num_correct = 0 
        for line in handle: # for each line
            num_chances += 1
            s = line.split()
            label = s[0] # Eng, ital, or french
            sentence = ' '.join(s[1:]).lower() # rest of sentence

            # create (n, 5c) matrix. Each row is a (1, 5c) one hot encoding vector
            # n is number of 5 seq characters in the sentence
            encode_mat = binarize(seq_chars(sentence))

            # accumulate pred for each (1,5c) vector
            pred = np.zeros(3)

            # create arbitrary label for each language
            # as long as you're consistent
            if label == "ENGLISH":
                y_test = np.array([0,1,0])
            elif label == "ITALIAN":
                y_test = np.array([1,0,0])
            else:
                y_test = np.array([0,0,1])

            num_rows = 0

            # for each 5 character encoder vector
            for row in range(len(encode_mat)):
                # get that row
                input_x = encode_mat[row,:].reshape(1,455)
                num_rows += 1

                # forward prop
                h_layer, y_pred = forward(input_x, W1, bias1, W2, bias2)
                # accumulate softmax
                pred += y_pred.reshape(3,)


            # AFTER a single sentence/row is done
            pred_avg = pred / num_rows  # avg prediction over all 5 character sequences
            loss = np.square(pred_avg - y_test).sum()# calculate loss over pred_avg
            loss_li.append(loss)
            pred_final = np.zeros(3)
            # get index of max probability of pred_avg to make final prediction
            pred_final[np.argmax(pred_avg)] = 1

            if np.all(pred_final - y_test == np.array([0,0,0])):
                num_correct += 1
            # do backpropogration using pred_avg over
            W1, bias1, W2, bias2 = backprop(pred_avg.reshape(3,1), y_test, h_layer, 
                                                input_x, W1, W2, bias1, bias2, eta=eta) 

    # after going through all lines
    accuracy = num_correct / num_chances
    accuracy_li.append(accuracy)


# In[62]:


import matplotlib.pyplot as plt
plt.plot(np.array(accuracy_li))
plt.show()


# In[63]:


plt.plot(np.array(loss_li))
plt.show()


# In[66]:


# np.random.seed(1)
filename = "languageIdentification.data/tiny_train" # 5528 lines
d = 100
eta = 0.1
input_dim = 455

W1 = np.random.uniform(size=(d, input_dim))
bias1 = np.random.uniform(size=[d,1])
W2 = np.random.uniform(size=(3,d))
bias2 = np.random.uniform(size=[3, 1])
eta = 0.1
accuracy_li = []
loss_li = []



# In[67]:


for i in range(3): # number of epochs where training data is 3 sentences
    with open(filename, 'r') as handle:
        num_chances = 0 # number of lines in text file
        num_correct = 0 
        for line in handle: # for each line
            num_chances += 1
            s = line.split()
            label = s[0] # Eng, ital, or french
            sentence = ' '.join(s[1:]).lower() # rest of sentence

            # create (n, 5c) matrix. Each row is a (1, 5c) one hot encoding vector
            # n is number of 5 seq characters in the sentence
            encode_mat = binarize(seq_chars(sentence))

            # accumulate pred for each (1,5c) vector
            pred = np.zeros(3)

            # create arbitrary label for each language
            # as long as you're consistent
            if label == "ENGLISH":
                y_test = np.array([0,1,0])
            elif label == "ITALIAN":
                y_test = np.array([1,0,0])
            else:
                y_test = np.array([0,0,1])

            num_rows = 0

            # for each 5 character encoder vector
            for row in range(len(encode_mat)):
                # get that row
                input_x = encode_mat[row,:].reshape(1,455)
                num_rows += 1

                # forward prop
                h_layer, y_pred = forward(input_x, W1, bias1, W2, bias2)
                # accumulate softmax
                pred += y_pred.reshape(3,)


            # AFTER a single sentence/row is done
            pred_avg = pred / num_rows  # avg prediction over all 5 character sequences
            loss = np.square(pred_avg - y_test).sum()# calculate loss over pred_avg
            loss_li.append(loss)
            pred_final = np.zeros(3)
            # get index of max probability of pred_avg to make final prediction
            pred_final[np.argmax(pred_avg)] = 1

            if np.all(pred_final - y_test == np.array([0,0,0])):
                num_correct += 1
            # do backpropogration using pred_avg over
            W1, bias1, W2, bias2 = backprop(pred_avg.reshape(3,1), y_test, h_layer, 
                                                input_x, W1, W2, bias1, bias2, eta=eta) 

    # after going through all lines
    accuracy = num_correct / num_chances
    accuracy_li.append(accuracy)



# In[71]:


plt.plot(np.array(accuracy_li))
plt.show()


# In[72]:


plt.plot(np.array(loss_li))
plt.show()

