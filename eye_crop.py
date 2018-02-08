from PIL import Image
import face_recognition
from os import listdir
from os.path import isfile, join

image_dir = 'robbie/'
fname = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    
for fil in range(len(fname)): 
    #image = face_recognition.load_image_file(image_dir + str(fname[fil]))
    image = Image.open(image_dir + str(fname[fil]))
    print("processed",fil)
    w = image.size[0]
    h = image.size[1]
    eye_image = image.crop((0, 0, w, h/2))
    #pil_image = Image.fromarray(eye_image)
    #pil_image.show()
    eye_image.save(image_dir + str(fname[fil]))
