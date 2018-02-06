from os import listdir
from os import rename
from os.path import isfile, join
onlyfiles = [f for f in listdir("./data") if isfile(join("./data", f))]

for fileName in onlyfiles:
	imageData = fileName.replace(".png","").split("-")
	if len(imageData) < 3:
		continue
	mouseX = float(imageData[1]) / 1440
	mouseY = float(imageData[2]) / 900
	rename("./data/"+fileName, "./data/"+imageData[0]+"-"+str(mouseX)+"-"+str(mouseY)+".png")
