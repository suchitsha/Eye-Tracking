import math

div_hor = 5
div_ver = 4

def convertToGlobalBoxNumber(filename):
	boxWidth = 1 / div_hor
	boxHeight = 1 / div_ver
	imageData = fileName.replace(".png","").split("-")
	mouseX = float(imageData[1])
	mouseY = float(imageData[2])
	print(mouseX)
	boxX = math.floor(mouseX / boxWidth)
	if mouseX >= 1.0:
		boxX = div_hor - 1
	boxY = math.floor(mouseY / boxHeight)
	if mouseY >= 1.0:
		boxY = div_ver - 1
	
	
	globalBoxNumber = (boxY * div_hor) + boxX
	return globalBoxNumber



print(convertToGlobalBoxNumber("1517870615.4922838-0.9-0.74"))
