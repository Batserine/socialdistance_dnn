# Importing packages
import cv2
import numpy as np
import math
import glob
from scipy.spatial import distance

model = 'yolo/yolov3.weights'
config = 'yolo/yolov3.cfg'
coco = 'yolo/coco.names'
net = cv2.dnn.readNetFromDarknet(config, model)
classes = []
with open(coco, "r") as f:
    classes = [line.strip() for line in f.readlines()]
   
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load Video

cap = cv2.VideoCapture('video.mp4')

count =0
count_pic=0
count_off=0

if (cap.isOpened()== False):
	print("Error opening video stream or file")
while(cap.isOpened()):
	ret, img = cap.read()
#	Exception handling
	if ret == True:
		assert not isinstance(img,type(None)), 'frame not found'
		(height,width,channels) = img.shape
		size = (width, height)
	
    # Detecting objects
		blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)

    # Showing informations on the screen
		class_ids = []
		confidences = []
		boxes = []

		for out in outs:
			for detection in out:
#				print("det",len(detection))
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
			            # Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)
	
			            # Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
	
					boxes.append([x, y, w, h])	
					confidences.append(float(confidence))
					class_ids.append(class_id)

		#NMS threshold
		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		count_ppl=0
		l=[]
		lf=[]
	
		#Adding bboxes around detections
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(classes[class_ids[i]])
				if label=='person':
					cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
					l =[]
					l.append(x)
					l.append(y)
					lf.append(l)				
					count_ppl+=1

		#Calculating pixel distance 
		off=0
		for i in range(len(lf)):
			for j in range(i+1,len(lf)):			
	
				#Pixel distance
				d=math.sqrt(((lf[j][1]-lf[i][1])**2)+((lf[j][0]-lf[i][0])**2))
#				print("Person",i+1,"- Person",j+1,"=",d)
#				print("Distance:",d)

				#Proximity is given to 50 for sorting out offenders			
				if d<50:
				 img = cv2.line(img, (lf[i][0],lf[i][1]), (lf[j][0],lf[j][1]), (0,0,255), 2)
				 off+=1
		count+=1   	
		if off>1:
			cv2.imwrite('offenders/img'+str(count_off)+'.jpg',img) # Saving frames in Offenders Folder
			count_off+=1
		

		#Storing frames into a folder.
		cv2.imwrite('frames/frame'+str(count)+'.jpg',img)
		img_array = []
		for filename in glob.glob('frames/*.jpg'):
			img1 = cv2.imread(filename)
			height, width, layers = img1.shape
			size = (width,height)
			img_array.append(img1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	else:
		print("Total no.of frames: "+str(count)+"  Total no.of Offending people: "+str(count_off))
		break		


out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
	out.write(img_array[i])
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
