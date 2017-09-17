import os
import cv2
import imutils
import random
from skimage.transform import rotate
#3 next ones used for the skewing
from skimage.transform import warp
from skimage import data
from skimage.transform import ProjectiveTransform
import numpy as np

#this class will rotate, translate, crop, add noise, skew, flip verticaly, flip horizontaly

class dataGenerationPipeline(object):

	def __init__(self, imageDirectory, fileType):
		#Load the images of type fileType contained in the directory and store them in a list
		self.inputDirectory = imageDirectory	
		self.inputImageList = []
		self.inputFileType = fileType	
		self.DBG = True

		if self.DBG:
			print("[DEBUG] : Retrieving {} images from directory {} "
								.format(self.inputFileType, self.inputDirectory))

		inputFileNames = [os.path.join(self.inputDirectory,f) 
						for f in os.listdir(self.inputDirectory)
						if f.endswith(str(fileType))]

		self.inputImageList = [cv2.imread(f) for f in inputFileNames] #store the images
		self.inputFilenames	= [f for f in os.listdir(self.inputDirectory) 
								if f.endswith(str(fileType))] #store the file names for latter user


		if self.DBG:
			print("[DEBUG] : Retrieved {} images".format(len(self.inputImageList)))

	def rotate(self):
		#This will create 5 more images for each input images, 
		#rotated by random degrees of the input directory 
		#and store them in the /rotated subdirectory of the input directory
		outputDirectory= os.path.join(self.inputDirectory,"rotated")

		if not os.path.exists(outputDirectory): #if directory does not exist create it
				os.makedirs(outputDirectory)
				if self.DBG:
					print("[DEBUG] : Created new directory {}".format(outputDirectory))

		for (fileName,image) in zip(self.inputFilenames, self.inputImageList):
			#usage of imutils librairy in oder to avoid the cropping of image when rotated
			#ref:http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
			for angle in (90,270): #rotate by 90 and 270 degrees
				rotated = imutils.rotate_bound(image, int(angle)) #This version leave black background
				#rotated = rotate(image, int(angle), resize=True, mode='edge', preserve_range = True)
				outputFilename = os.path.join(outputDirectory, "rot"+ str(int(angle)) +"_" + str(fileName))
				cv2.imwrite (outputFilename, rotated) #store rotated image 
				#if self.DBG:
					#print("[DEBUG] : Saved Image {}".format(outputFilename))

		if self.DBG:
			numberImages= len(os.listdir(outputDirectory))
			print("[DEBUG] : Saved {} images in {} ".format(numberImages, outputDirectory))

	def flip(self, horizontaly = False, verticaly = False):
		#flip the images ad save them to the "flipped" sub-directory
		outputDirectory=os.path.join(self.inputDirectory,"flipped")

		if not os.path.exists(outputDirectory): #if directory does not exist create it
				os.makedirs(outputDirectory)
				if self.DBG:
					print("[DEBUG] : Created new directory {}".format(outputDirectory))

		for (fileName,image) in zip(self.inputFilenames, self.inputImageList):
			if horizontaly and not verticaly: #flip horizontaly
				flipped = cv2.flip(image, 1)
				outputFilename = os.path.join(outputDirectory, "horFlip_"+str(fileName))
				cv2.imwrite (outputFilename, flipped) #store flipped image

			if verticaly and not horizontaly: #flip vertically
				flipped = cv2.flip(image, 0)
				outputFilename = os.path.join(outputDirectory,"verFlip_"+str(fileName))
				cv2.imwrite (outputFilename, flipped) #store flipped image
			
			if verticaly and horizontaly: #flip bothy
				flipped = cv2.flip(image, -1)
				outputFilename = os.path.join(outputDirectory, "horVerFlip_"+str(fileName))
				cv2.imwrite (outputFilename, flipped) #store flipped image


		if self.DBG:
			numberImages= len(os.listdir(outputDirectory))
			print("[DEBUG] : Saved {} images in {} ".format(numberImages, outputDirectory))

	def addnoise(self, type='gauss'):
		#will create on disk images blurred with gausian noise
		outputDirectory=os.path.join(self.inputDirectory,"noise")

		if not os.path.exists(outputDirectory): #if directory does not exist create it
				os.makedirs(outputDirectory)
				if self.DBG:
					print("[DEBUG] : Created new directory {}".format(outputDirectory))

		for (fileName,image) in zip(self.inputFilenames, self.inputImageList):
			#https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv/30609854
			if type == 'gauss':
				row,col,ch= image.shape
				mean = 0
				var = 122
				sigma = var**0.5
				gauss = np.random.normal(mean,sigma,(row,col,ch))
				gauss = gauss.reshape(row,col,ch)
				noisy = image + gauss #numpy will perfrorm modulo additions/substractions
				outputFilename = os.path.join(outputDirectory, "gauss_"+str(fileName))
				cv2.imwrite (outputFilename, noisy) #store noisy image

			elif type == 's&p':
				row,col,ch = image.shape
				s_vs_p = 0.5
				amount = 0.004
				out = np.copy(image)
				# Salt mode
				num_salt = np.ceil(amount * image.size * s_vs_p)
				coords = [np.random.randint(0, i - 1, int(num_salt))
					for i in image.shape]
				out[coords] = 255 #add some white randomly

				# Pepper mode
				num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
				coords = [np.random.randint(0, i - 1, int(num_pepper))
					for i in image.shape]
				out[coords] = 0 #add some black randomly

				outputFilename = os.path.join(outputDirectory, "s&p_"+str(fileName))
				cv2.imwrite (outputFilename, out) #store s&p image


		if self.DBG:
			numberImages= len(os.listdir(outputDirectory))
			print("[DEBUG] : Saved {} images in {} ".format(numberImages, outputDirectory))

	def blur(self, size=(5,5)):
		#blur the image using a gaussian filter of size "size"

		outputDirectory=os.path.join(self.inputDirectory,"blurred")

		if not os.path.exists(outputDirectory): #if directory does not exist create it
				os.makedirs(outputDirectory)
				if self.DBG:
					print("[DEBUG] : Created new directory {}".format(outputDirectory))

		for (fileName,image) in zip(self.inputFilenames, self.inputImageList):
			blurred = cv2.GaussianBlur(image, size, 0) #3rd param tells cv2 to compute sigma according to filter size
			outputFilename = os.path.join(outputDirectory,"blur_"+str(fileName))
			cv2.imwrite (outputFilename, blurred) #store blurred image 

		if self.DBG:
			numberImages= len(os.listdir(outputDirectory))
			print("[DEBUG] : Saved {} images in {} ".format(numberImages, outputDirectory))

	def skew(self):
		#return 5 versions of the original image skewed in 5 different random ways, 
		#http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.ProjectiveTransform

		outputDirectory=self.initializeDirectory("skewed") #prepare the output directory

		for (fileName,image) in zip(self.inputFilenames, self.inputImageList):
			(imageLength, imageWidth,_) = image.shape
			skewExtend = int(max(imageLength, imageWidth) * 0.20) #Will distord the image by max 20% of max length, width

			for nbNewImages in range(5): #generate 5 skewed images
				
				d = skewExtend #distorsion applied is proportionnal to the parameter
				#define the distortions to apply
				topLeftTopShift = random.uniform(-d, d)
				topLeftLeftShift = random.uniform(-d, d)
				bottomLeftBottomShift = random.uniform(-d, d)
				bottomLeftLeftShift = random.uniform(-d, d)
				topRightTopShift = random.uniform(-d, d)
				topRightRightShift = random.uniform(-d, d)
				bottomRightBottomShift = random.uniform(-d, d)
				bottomRightRightShift = random.uniform(-d, d)

				#enable the projective transform
				transform = ProjectiveTransform()

				#tear the image
				transform.estimate(np.array((
										(topLeftLeftShift , topLeftTopShift),
										(bottomLeftLeftShift, imageWidth- bottomLeftBottomShift),
										(imageLength - bottomRightRightShift, imageWidth - bottomRightBottomShift),
										(imageLength- topRightRightShift, topRightTopShift))),
									np.array((
										(0,0),
										(0, imageWidth),
										(imageLength, imageWidth),
										(imageLength,0)))
									)
				#apply the skew
				skewed = warp(image, transform, mode='edge')
				skewed = skewed*255
				outputFilename = os.path.join(outputDirectory,"skewed"+str(nbNewImages)+"_"+str(fileName))
				cv2.imwrite (outputFilename, skewed) #store skewed image

		if self.DBG:
			numberImages= len(os.listdir(outputDirectory))
			print("[DEBUG] : Saved {} images in {} ".format(numberImages, outputDirectory))

	def crop(self):
		#return 6 versions of the original image cropped  

		outputDirectory=self.initializeDirectory("cropped") #prepare the output directory

		for (fileName,image) in zip(self.inputFilenames, self.inputImageList):
			(imageLength, imageWidth,_) = image.shape
			width25Percent = int(0.25*imageWidth)
			width75Percent = int(0.75*imageWidth)
			length25Percent = int(0.25*imageLength)
			length75Percent = int(0.75*imageLength)


			width33Percent = int(0.33*imageWidth)
			width66Percent = int(0.66*imageWidth)
			length33Percent = int(0.33*imageLength)
			length66Percent = int(0.66*imageLength)

			cropped=[]

			cropped.append(image[length25Percent:length75Percent ,  0:imageWidth]) #vertical layer 0.25 till 0.75 length, full width
			cropped.append(image[0:imageLength ,  width25Percent:width75Percent]) #horizontal layer 0.25 till 0.75 width, full length
			cropped.append(image[0:length66Percent ,  0:width66Percent]) #window 0 --> 0.66 width, 0 --> 0.66 length
			cropped.append(image[0:length66Percent ,  width33Percent:imageWidth]) #window 0.33 --> 1 width, 0 --> 0.66 length
			cropped.append(image[length33Percent:imageLength , 0:width66Percent]) #window 0 --> 0.66 width, 0.33 --> 1 length
			cropped.append(image[length33Percent:imageLength , width33Percent:imageWidth]) #window 0.33 --> 1 width, 0.33 --> 1 length

			for (i,image) in enumerate(cropped):
				outputFilename = os.path.join(outputDirectory,"cropped"+str(i)+"_"+str(fileName))
				cv2.imwrite (outputFilename, cropped[i]) #store cropped image

		if self.DBG:
			numberImages= len(os.listdir(outputDirectory))
			print("[DEBUG] : Saved {} images in {} ".format(numberImages, outputDirectory))


	def initializeDirectory(self, directoryName):

		#will create the output directory if needed and return its name
		outputDirectory=os.path.join(self.inputDirectory,str(directoryName)) #save them in the new directory

		if not os.path.exists(outputDirectory): #if directory does not exist create it
				os.makedirs(outputDirectory)
				if self.DBG:
					print("[DEBUG] : Created new directory {}".format(outputDirectory))
		return outputDirectory




			




