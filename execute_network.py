import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
from random import randint
import _pickle as pickle
import cv2
import itertools
import math
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from os import listdir
from os.path import isfile, join


#def init(self):
div_hor = 5 #50 # 20
div_ver = 4
out_size = div_hor*div_ver
lr = 0.000001 #4 0s
epochs = 20#0
batch_size = 100#0

image_dir = 'data/'
test_dir = 'test/'
#s_direct = 'model/model.ckpt'
s_direct = 'model_backup/model.ckpt-315800'
#out_dir = '/home_local/shar_sc/cnn_result/'
#    pass

def convertToGlobalBoxNumber(file_name):
	boxWidth = 1 / div_hor
	boxHeight = 1 / div_ver
	imageData = file_name.replace(".png","").split("-")
	mouseX = float(imageData[1])
	mouseY = float(imageData[2])
	#print(mouseX)
	boxX = math.floor(mouseX / boxWidth)
	if mouseX >= 1.0:
		boxX = div_hor - 1
	boxY = math.floor(mouseY / boxHeight)
	if mouseY >= 1.0:
		boxY = div_ver - 1
	
	
	globalBoxNumber = (boxY * div_hor) + boxX
	return globalBoxNumber

    
def execute():
    #mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
    #img = mnist.train.images[2]
    #plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
    learning_rate = lr#0.001
    # Input and target placeholders
    inputs_ = tf.placeholder(tf.float32, (None, 240,100,1), name="input")
    targets_ = tf.placeholder(tf.float32, (None, out_size), name="target")

    ### Encoder
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 240x100x64
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 120x50x64
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=48, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 120x50x48
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 60x25x48
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 60x25x32
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 30x13x32

    conv3_1 = tf.layers.conv2d(inputs=encoded, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    # Now 30x13x8
    encoded2 = tf.layers.max_pooling2d(conv3_1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 15x7x8
        
    # Dense Layer
    encoded2_flat = tf.reshape(encoded2, [-1, 15 * 7 * 8])
    dense = tf.layers.dense(inputs=encoded2_flat, units=512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense, units=512, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu)
    #TODO try dropout
    #dropout = tf.layers.dropout(inputs=dense3, rate=0.1)#, training=mode == learn.ModeKeys.TRAIN)
    # Logits Layer
    #logits = tf.layers.dense(inputs=dropout, units=20)
    uni = out_size
    logits = tf.layers.dense(inputs=dense3, units=uni)
    out_shape = tf.reshape(logits,[-1,uni] )    
    # Pass logits through sigmoid to get reconstructed image
    #decoded = tf.nn.sigmoid(logits)
    loss = None
    train_op = None

    # Pass logits through sigmoid and calculate the cross-entropy loss
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=out_shape)
    
    # Get cost and define the optimizer
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #init regressor
    #regressor = init_dnn()

    #direct = image_dir + 'direc.p'
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    '''
    
    fname = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    #print(fname)
    objs = []
    # for testing
    images_test = []
    objs_test = []
    objs_test_tensor = []
    step = 0
    #read images
    #print("asahsh", fname)
    for e  in range(epochs):
        for i in range(len(fname)):            
            out_val = convertToGlobalBoxNumber(str(fname[i]))
            print("out_val from file",out_val)
            objs.append(out_val)
            img_direct = image_dir + str(fname[i])#image_dir + 'fig_' + str(i) + '.png'
            img = cv2.imread(img_direct,0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,dsize=(240,100) , interpolation = cv2.INTER_CUBIC) #(128,128) , interpolation = cv2.INTER_CUBIC)
            #cv2.imshow('image',img)
            #Numpy array
            np_image_data = np.asarray(img)
            np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            np_final = np.expand_dims(np_image_data,axis=0)
            np_final = np_final.reshape((240,100,-1))   
            images_in = []
            images_in.append(np_final)
            #images_in.append(np_final)
            inp = np.asarray(out_val)
            inp = inp.astype(np.int32)#np.float32)#np.int32    
            #inp = inp.reshape([-1,20])    
            
            #inp = tf.one_hot(indices = inp, depth = 4, on_value = 1.0, off_value = 0.0, axis = -1) 
            print("len variable images_in:", len(images_in))   
            print("variable inp",inp)
            #inp = inp.reshape(len(inp), 1)
            onehot_encoder = OneHotEncoder(n_values=out_size, sparse=False)
            inp = onehot_encoder.fit_transform(inp)
            print("training output",inp)
            
            if len(inp[0]) != out_size:
                print("size of labels not correct:", inp)#len(inp)
                continue
            inp = inp.reshape([-1,out_size])    
            
            if i < 0:
                if (e == 0):
                    images_test.append(np_final)
                    objs_test.append(objs[i])
                    objs_test_tensor.append(inp)
            else:     
                #print("asaaasa",images_in)
                #print("asahdsj",inp)   
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: images_in, targets_: inp})
                print("Epoch: {}/{}/{}".format(i+1,e+1, epochs), "Training loss: {:.4f}".format(batch_cost))
                step = step +1
        
        save_path = saver.save(sess,s_direct,global_step=step)
        print("Model saved in file: %s" % save_path)
    '''
    
    fname1 = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
    lab_arr = []
    expected_box = []
    
    #restore model
    saver.restore(sess, s_direct) #("/tmp/model.ckpt")
    print("Model restored.")
    
    for fil in range(len(fname1)):            
            out_val1 = convertToGlobalBoxNumber(str(fname1[fil]))
            #print("expected_output",out_val1)
            expected_box.append(out_val1)
            img_direct1 = test_dir + str(fname1[fil])#image_dir + 'fig_' + str(i) + '.png'
            img1 = cv2.imread(img_direct1,0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img1 = cv2.resize(img1,dsize=(240,100) , interpolation = cv2.INTER_CUBIC) #(128,128) , interpolation = cv2.INTER_CUBIC)
            #cv2.imshow('image',img)
            #Numpy array
            np_image_data1 = np.asarray(img1)
            np_image_data1 = cv2.normalize(np_image_data1.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            np_final1 = np.expand_dims(np_image_data1,axis=0)
            np_final1 = np_final1.reshape((240,100,-1)) 
            images_in1 = []
            images_in1.append(np_final1)
            #print("saehadhah", len(images_in1))
            reconstructed = sess.run(logits, feed_dict={inputs_: images_in1})
            reconstructed = np.asarray(reconstructed).reshape([-1,out_size])
            print ("reconstructed:", reconstructed)            
            max_index = argmax(reconstructed)
            lab_arr.append(max_index)
    print("results:", lab_arr)    
    print("expected_box",expected_box)        
    #predictions = list(itertools.islice(reconstructed, 1))
    #predictions = np.asarray(predictions).reshape([20,4]) 
    #print("Predictions: {}".format(str(len(predictions))))
    
    #print("labels:", objs_test)
    sess.close()  
    return

execute()
