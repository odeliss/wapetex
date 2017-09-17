import sys
import os
from datagenerationpipeline import dataGenerationPipeline
import numpy as np
import json

#The directory of images is assumed to structured in subdirectories named by consecutive integers, 
#starting at 0: eg c:johnDoe/images/watches/0 represent category 0 of the watches images
#For this particular case, the new rotated images will be stored in c:johnDoe/images/watches/0/rotated subdirectory 
#A particular subdirectory is created for each transformation in the <<categoryNb>> subdirectory

conf= json.load(open('conf.json'))

imageDirectory = conf["imageDirectory"] #will use the directory in the conf file
nbCategories = int(conf["nbCategories"]) #this is the number of sub-directories in the images directory
imageFormat = conf["imageFormat"] #only images of this format will be loaded

for subdirectory in range(0, nbCategories): 
	imageSubdirectory = os.path.join(imageDirectory,str(subdirectory)) 
	fileType=imageFormat  #all images are assumed to be JPG. 
	pipeline=dataGenerationPipeline(imageSubdirectory, fileType)
	pipeline.rotate()
	pipeline.flip(horizontaly = True)
	pipeline.flip(verticaly = True)
	pipeline.flip(horizontaly = True, verticaly = True)
	pipeline.addnoise(type='gauss')
	pipeline.addnoise(type='s&p')
	pipeline.blur() #blur with a 5x5 filter
	pipeline.skew()
	pipeline.crop()