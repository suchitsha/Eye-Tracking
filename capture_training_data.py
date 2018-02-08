from sneakysnek.recorder import Recorder
from sneakysnek.mouse_event import MouseEvents
import cv2
import time

camera = cv2.VideoCapture(0)

def takePicture(clickEvent):
	if clickEvent.event == MouseEvents.CLICK and clickEvent.direction == "DOWN":
		print("clicked at: " + str(clickEvent.x) + ", " + str(clickEvent.y))
		return_value, image = camera.read()
		# resized_image = cv2.resize(image, (500, 281)) 
		cv2.imwrite('./suchit/'+str(time.time()) + '-' + str(clickEvent.x) + '-' + str(clickEvent.y) + '.png', image)


recorder = Recorder.record(takePicture)  # Replace print with any callback that accepts an 'event' arg
# Some blocking code in your main thread...

while True:
	True
