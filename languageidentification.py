
# coding: utf-8

# In[1]:


from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from sklearn.preprocessing import LabelBinarizer
import string
import numpy as np
import math
import time
import sys
from io import open



def sigmoid_array(x):
    # input: array
    # output: sigmoid applied to each value of input array
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # input: array
    # output: softmax of array
    u"""Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))



def seq_chars(s, num_chars=5):
    # input: string, s
    # output: list of 5 sequential characters of string from beginning to end
    n = len(s)
    return [s[i:(i+num_chars)] for i in xrange(n-4)]

def binarize(seq_str):
    # input: sequence of characters
    # output: concatenated one hot encoding of each character
        # w/ dimension (5c, 1)
    nseq = len(seq_str)
    if len(seq_str) == 0:
        return np.zeros(shape = (1,input_dim))
    # for char in seq_str[0]:
    #     if char not in alphab:
    #         print char, u'not in alphabet'
    return np.array([le.transform(list(seq_str[i])) for i in xrange(nseq)]).reshape(nseq, input_dim)




def grad_l_wrt_y(y_pred, y_test):
    return y_pred - y_test #(3,1)


# $\nabla_{b^{2}} L = \nabla_{y^{'}}L = \sum_{i} \frac{\delta{L}}{\delta{y_{i}}} y_{i}(\delta{ij} - y_{j}) = \sum_{i} (y_{i} - \hat{y_{i}}) y_{i}(\delta_{ij} - y_{j})$


def grad_l_wrt_b2(y_pred, y_test):
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


def grad_l_wrt_w2(grad_b2, hidden_layer):
    # take grad wrt b^2 and mult by hidden layer
    return grad_b2.dot(hidden_layer.T) #(3,1)*(1,d) is (3,d)



# $\nabla_{h}L = W^{2T} \nabla_{y^{'}}L = W^{2T} \nabla_{b^{2}}L$


def grad_l_wrt_h(grad_b2, W2):
    return np.dot(W2.T, grad_b2) #(d,3)*(3,1) is (d,1)


# $\frac{\delta{L}}{\delta{h^{'}_{i}}} = \frac{\delta{L}}{\delta{h_{i}}} h_{i}(1-h_{i})$ ie multiply each elt in $\nabla_{h}L$ by $h_{i}(1-h_{i})$



def grad_l_wrt_h_tilde(grad_h, h_layer):
    # does element wise multiplication
    return grad_h * ((h_layer) * (1-h_layer)) # (d,1)(element mult)(d,1) is (d,1)

#     vec = []
#     for i in range(len(h_layer)):
# #         print("mult:", grad_h[i], (h_layer[i] * (1-h_layer[i])))
#         vec.append(grad_h[i] * (h_layer[i] * (1-h_layer[i])))

#     return np.array(vec)



# $\nabla_{w^{1}} L = (\nabla_{h^{'}}L)(x^{T})$



def grad_l_wrt_w1(grad_h_tilde, input_x):
    return np.dot(grad_h_tilde, input_x) #(d,1)*(1,5c) is (d,5c)



# $\nabla_{b^{1}} L = (\nabla_{h^{'}}L)$


def grad_l_wrt_b1(grad_h_tilde):
    return grad_h_tilde




def create_input_matrix(filename):
    with open(filename, u'r') as handle:
        y_final = []
        final_mat = []
        for line in handle: # for each line
            y_test = []


            s = line.split()
            label = s[0] # Eng, ital, or french
            sentence = u' '.join(s[1:]).lower() # rest of sentence

            # create (n, 5c) matrix. Each row is a (1, 5c) one hot encoding vector
            # n is number of 5 seq characters in the sentence
            encode_mat = binarize(seq_chars(sentence))
#             final_mat = np.vstack([final_mat, encode_mat])
            final_mat.append(encode_mat)


            # for each 5 character encoder vector
            for row in xrange(len(encode_mat)):
                    # get that row
                if label == u"ENGLISH":
                    y_test.append(1)
                elif label == u"ITALIAN":
                    y_test.append(2)
                else:
                    y_test.append(3)
            y_final.append(y_test)
    # turn list of numpy arrays (for each sentence) into one huge numpy array
    fin = np.vstack(final_mat)
    # turn list of labels (for each sentence) into one long numpy vector
    fin_lab = np.concatenate([np.array(i) for i in y_final])
    # combine them into a silly thing called fin_fin
    fin_fin = np.hstack([fin, fin_lab.reshape(len(fin_lab),1)])
    return fin_fin



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


def backprop(y_pred, y_test, h_layer, input_x, W1, bias1, W2, bias2, eta=.1):   
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

    W1 = W1 - eta * grad_l_w1
    W2 = W2 - eta * grad_l_w2
    bias2 = bias2 - eta * grad_l_b2
    bias1 = bias1 - eta * grad_l_b1

    return W1, bias1, W2, bias2


def train_nn(train_data, W1, bias1, W2, bias2):
    # get the last column vector
    train_labels = train_data[:,-1]
    train_data = np.delete(train_data, (-1), axis=1)

    # for each 5 character encoder vector
    for ind in xrange(len(train_data)):
        # create arbitrary label for each language
        if train_labels[ind] == 1: # english
            y_test = np.array([0,1,0])
        elif train_labels[ind] == 2: # italian
            y_test = np.array([1,0,0])
        else:
            y_test = np.array([0,0,1])


        # get that row
        input_x = train_data[ind, :].reshape(1,input_dim)
        
        # forward prop
        h_layer, y_pred = forward(input_x, W1, bias1, W2, bias2)
        # backwards prop
        W1, bias1, W2, bias2 = backprop(y_pred.reshape(3,1), y_test, h_layer, 
                                        input_x, W1, bias1, W2, bias2, eta=eta)

    return W1, bias1, W2, bias2


def test_nn(filename, W1, bias1, W2, bias2, file2=None):
    loss_li = []
    counter = 0
    code = u'utf-8'
    pred_list = []
    # if testing on test set
    if file2 is not None:
        code = u'latin-1'
        with open(file2, u'r') as f: # the test solutions file
            lines = f.read().lower().splitlines()

    with open(filename, u'r', encoding=code) as handle:
        num_chances = 0 # number of lines in text file
        num_correct = 0 
        for line in handle: # for each line
            num_chances += 1
            # if testing on training or dev
            if line.split()[0] in [u'ENGLISH', u'ITALIAN', u'FRENCH']:
                s = line.split()
                label = s[0].lower() # Eng, ital, or french
                sentence = u' '.join(s[1:]).lower() # rest of sentence
            else:
                sentence = line.lower()
                label = lines[counter]
                counter += 1

            # create (n, 5c) matrix. Each row is a (1, 5c) one hot encoding vector
            # n is number of 5 seq characters in the sentence
            encode_mat = binarize(seq_chars(sentence))

            # to accumulate pred for each (1,5c) vector
            pred = np.zeros(3)

            # create arbitrary label for each language
            # as long as you're consistent
            if u"english" in label:
                y_test = np.array([0,1,0])
            elif u'italian' in label:
                y_test = np.array([1,0,0])
            else:
                y_test = np.array([0,0,1])

            num_rows = 0

            # for each 5 character encoder vector
            for row in xrange(len(encode_mat)):
                # get that row
                input_x = encode_mat[row,:].reshape(1,input_dim)
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
            ind = np.argmax(pred_avg)
            pred_final[ind] = 1
            if ind == 1:
                rslt = u"English"
            elif ind == 0:
                rslt = u"Italian"
            else:
                rslt = u"French"
            pred_list.append(rslt)

            if np.all(pred_final - y_test == np.array([0,0,0])):
                num_correct += 1

    # after going through all lines
    accuracy = num_correct / num_chances
    return accuracy, pred_list


if __name__ == u"__main__":
    start = time.time()
    train = sys.argv[1]
    dev = sys.argv[2]
    test = sys.argv[3]

    txt = u' '
    with open(train, u'r') as f:
        txt += f.read()

    alphab = list(set(list(txt.lower())))
    input_dim = (len(alphab)) * 5


    le = LabelBinarizer()
    le.fit(alphab)

    ######### PART A ##########
    d = 100
    eta = 0.1

    W1 = np.random.uniform(size=(d, input_dim))
    bias1 = np.random.uniform(size=[d,1])
    W2 = np.random.uniform(size=(3,d))
    bias2 = np.random.uniform(size=[3, 1])
    print u'creating input matrix'
    tiny_train_mat = create_input_matrix(train)
    print u'created input_matrix'
    print time.time() - start, u'secs'


    acc_li_train = []
    acc_li_dev = []
    loss_acum = []
    # train 3 times
    print u'time to train part a'
    for i in xrange(4):
        print u'a: trained {} times'.format(i)
        print time.time() - start, u'secs'
        # get test before training, and after each train
        accuracy_t, _ = test_nn(train, W1, bias1, W2, bias2)
        accuracy_d, _ = test_nn(dev, W1, bias1, W2, bias2)


        acc_li_train.append(accuracy_t)
        acc_li_dev.append(accuracy_d)

        np.random.shuffle(tiny_train_mat)
        W1, bias1, W2, bias2 = train_nn(tiny_train_mat, W1, bias1, W2, bias2)




    # print('creating figure ')
    # fig = plt.figure()
    # plt.plot(acc_li_train)
    # plt.plot(acc_li_dev)
    # plt.legend(['train accuracy', 'dev accuracy'], loc='upper left')
    # plt.xlabel('epoch', fontsize=18)
    # plt.ylabel('accuracy', fontsize=16)
    # plt.xticks([0,1,2,3])
    # print('saving figure')
    # fig.savefig('accuracy.png')

    print u'part a training accuracies', acc_li_train
    print u'part a dev accuracies', acc_li_dev


    # test on the testing data
    accuracy_test, predictions = test_nn(test, W1, bias1, W2,
                                        bias2, u"languageIdentification.data/test_solutions")
    counter = 0
    outfile = open(u'languageIdentificationPart1.output', u'w')
    with open(test, encoding = u'latin-1') as f:
        for line in f:
            outfile.write(line[:-1] + u' ' + predictions[counter] + u'\n')
            counter+=1
    outfile.close()
    print u'created languageIdentificationPart1.output'
    print u'part a testing accuracy:', accuracy_test

    ######### part b ##########

    print u'part b'
    print time.time() - start, u'secs'

    set1 = (250, .75)
    set2 = (25, .75)
    set3 = (25, .05)
    set4 = (100, .05)
    set5 = (3, .075)

    params_list = [set1, set2, set3, set4, set5]
    dict_params = {}




    for params in params_list:
        d, eta = params
        W1 = np.random.uniform(size=(d, input_dim))
        bias1 = np.random.uniform(size=[d,1])
        W2 = np.random.uniform(size=(3,d))
        bias2 = np.random.uniform(size=[3, 1])

        # train 3 times
        for i in xrange(3):
            # train
            np.random.shuffle(tiny_train_mat)
            W1, bias1, W2, bias2 = train_nn(tiny_train_mat, W1, bias1, W2, bias2)


        # test on dev
        accuracy_d, _ = test_nn(dev, W1, bias1, W2, bias2)
        
        dict_params[params] = [accuracy_d, W1, bias1, W2, bias2]
        print u'b: trained {} parameters'.format(params)
        print time.time() - start, u'secs'




    li_keys = list(dict_params.keys())
    best_key = li_keys[0]
    for key in li_keys:
        print key, u":", dict_params[key][0]
        if dict_params[best_key][0] < dict_params[key][0]:
            best_key = key
    print u'best params:', best_key




    # get the weights and biases from the best hyperparameters
    W1_best, bias1_best, W2_best, bias2_best = dict_params[best_key][1:]
    accuracy_test, _ = test_nn(test, W1_best, bias1_best, W2_best, bias2_best,
                               u"languageIdentification.data/test_solutions")
    print u'part b test accuracy:', accuracy_test
    print time.time() - start, u'secs'

