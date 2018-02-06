import math

xNumberOfBlocks = 5
yNumberOfBlocks = 4

def convertToGlobalBoxNumber(filename):
	boxWidth = 1 / xNumberOfBlocks
	boxHeight = 1 / yNumberOfBlocks
	imageData = fileName.replace(".png","").split("-")
	mouseX = float(imageData[1])
	mouseY = float(imageData[2])
	print(mouseX)
	boxX = math.floor(mouseX / boxWidth)
	if mouseX >= 1.0:
		boxX = xNumberOfBlocks - 1
	boxY = math.floor(mouseY / boxHeight)
	if mouseY >= 1.0:
		boxY = yNumberOfBlocks - 1
	
	
	globalBoxNumber = (boxY * xNumberOfBlocks) + boxX
	return globalBoxNumber



print(convertToGlobalBoxNumber("1517870615.4922838-0.9-0.74"))