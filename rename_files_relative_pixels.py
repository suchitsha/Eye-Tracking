from os import listdir
from os import rename
from os.path import isfile, join
onlyfiles = [f for f in listdir("./robbie") if isfile(join("./robbie", f))]

for fileName in onlyfiles:
	if fileName.startswith("._"):
		continue
	imageData = fileName.replace(".png","").split("-")
	if len(imageData) < 3:
		continue
	mouseX = float(imageData[1]) / 1440
	mouseY = float(imageData[2]) / 900
	rename("./robbie/"+fileName, "./robbie/"+imageData[0]+"-"+str(mouseX)+"-"+str(mouseY)+".png")
