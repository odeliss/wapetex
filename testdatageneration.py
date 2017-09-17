from datagenerationpipeline import dataGenerationPipeline

imageDirectory = "\\\\192.168.1.37\\Multimedia\\datasets\\test\\watches_categories\\1"

fileType=".jpg"
pipeline=dataGenerationPipeline(imageDirectory, fileType)
print("[INFO]: There should be 200 images in directory 1")


pipeline.rotate()
#assert there are 1000 flipped images in directory rotated

pipeline.flip(horizontaly = True)
#assert there are 200 flipped images in directory horFlip

pipeline.flip(verticaly = True)
#assert there are 200 flipped images in directory verFlip

pipeline.flip(horizontaly = True, verticaly = True)
#assert there are 200 flipped images in directory horverFlip


pipeline.skew()