from PIL import Image
import face_recognition
from os import listdir
from os.path import isfile, join

image_dir = 'suchit/'
fname = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    
for fil in range(len(fname)): 
    image = face_recognition.load_image_file(image_dir + str(fname[fil]))
    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    print("processed",fil)
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    #if len(face_locations) == 0:
    #    continue
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
    #pil_image.show()
    pil_image.save(image_dir + str(fname[fil]))