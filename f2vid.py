#Steps:
# 1.Read each image using cv2.imread()
# 2.Initialize the video writer using cv2.VideoWriter()
# 3.Save the frames to a video file using cv2.VideoWriter.write()
# 4.Wait for keyboard button press using cv2.waitKey()
# 5.Release the VideoWriter using cv2.VideoWriter.release()
# 6.Exit window and destroy all windows using cv2.destroyAllWindows()

import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
	# Define the codec and create VideoWriter object 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    pathIn= 'frames/'
    pathOut = 'output.avi'
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()

