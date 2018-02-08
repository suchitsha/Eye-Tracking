from os import listdir
from os import rename
from os.path import isfile, join
dir_name = "./suchit11/"
onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

for fileName in onlyfiles:
	if fileName.startswith("._"):
		continue
	imageData = fileName.replace(".png","").split("-")
	if len(imageData) < 3:
		continue
	mouseX = float(imageData[1]) / 1440
	mouseY = float(imageData[2]) / 900
	rename(dir_name+str(fileName),dir_name+imageData[0]+"-"+str(mouseX)+"-"+str(mouseY)+".png")

