import numpy as np
import tensorflow as tf
import cv2
from numpy import argmax
import face_recognition
from PIL import Image


camera = cv2.VideoCapture(0)

#def init(self):
div_hor = 5 #50 # 20
div_ver = 4
out_size = div_hor*div_ver


s_direct = 'model/model.ckpt-312642'

# Input and target placeholders
inputs = tf.placeholder(tf.float32, (None, 240,100,1), name="input")
targets_ = tf.placeholder(tf.float32, (None, out_size), name="target")

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
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

logits = tf.layers.dense(inputs=dense3, units=out_size)  



# Add ops to save and restore all the variables.
saver = tf.train.Saver()


sess = tf.Session()

#restore model
saver.restore(sess, s_direct) #("/tmp/model.ckpt")
print("Model restored.")


sess.run(tf.global_variables_initializer())

while True:
    return_value, image = camera.read()
    face_locations = face_recognition.face_locations(image)
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    if len(face_locations) > 0:
        firstFoundFace = face_locations[0]
        top, right, bottom, left = firstFoundFace
        face_image = image[top:bottom, left:right]
        face_image = Image.fromarray(face_image)
        w = face_image.size[0]
        h = face_image.size[1]
        eye_image = face_image.crop((0, 0, w, h/2))
        # NOW EXECUTE THE MODEL HERE AND GET PREDICTION WHERE YOU ARE LOOKING AT

        # convert PIL image to CV2 image
        eye_image = cv2.cvtColor(np.array(eye_image), cv2.COLOR_RGB2GRAY)
        eye_image = cv2.resize(eye_image,dsize=(240,100) , interpolation = cv2.INTER_CUBIC)
        # cv2.imwrite("./eye/imag.png", eye_image)
        np_image_data = np.asarray(eye_image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data,axis=0)
        np_final = np_final.reshape((240,100,-1)) 
        images_in = [np_final]
        reconstructed = sess.run(logits, feed_dict={inputs: images_in})
        reconstructed = np.asarray(reconstructed).reshape([-1,out_size])
        max_index = argmax(reconstructed)
        print ("reconstructed:", max_index)       


sess.close()
