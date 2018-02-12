import cv2
import time
from PIL import Image
import time
import face_recognition
import tensorflow as tf
import numpy as np

camera = cv2.VideoCapture(0)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'model/model.ckpt-315800')

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
		eye_image = cv2.cvtColor(np.array(eye_image), cv2.COLOR_RGB2BGR)
		eye_image = cv2.resize(eye_image, (240,100))
		np_image_data = np.asarray(eye_image)
		np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
		np_final = np.expand_dims(np_image_data,axis=0)
		np_final = np_final.reshape((240,100,-1)) 
		images_in = [np_final]
		reconstructed = sess.run(logits, feed_dict={inputs_: images_in1})
		reconstructed = np.asarray(reconstructed).reshape([-1,out_size])
		print ("reconstructed:", reconstructed)            
