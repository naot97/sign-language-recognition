import cv2
import numpy as np
import copy
import math
import utils
import time


from keras.models import load_model

# parameters
cap_region_x_begin=0.7  # start point/total width
cap_region_y_end=0.5  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured

#count
count = 0
def mapCharacter(n):
	return {
		0: 'Nothing',1: 'A',2: 'B',3: 'C',4: 'D',5: 'Đ',6: 'E',7: 'G',8: 'H',9: 'I',
								10: 'K',11: 'L',12: 'M',13: 'N',14: 'O',15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',
								20: 'U',21: 'V',22: 'X',23: 'Y',24: '^',25: 'Ư',

				}.get(n,'null')

def printThreshold(thr):
	print("! Changed threshold to "+str(thr))


def removeBG(frame):
	fgmask = bgModel.apply(frame,learningRate=learningRate)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

	kernel = np.ones((3, 3), np.uint8)
	#fgmask = cv2.erode(fgmask, kernel, iterations=1)
	res = cv2.bitwise_and(frame, frame, mask=fgmask)
	return res


bow=['','','']
index=0
#def makeSure(index,char):
#	bow[index]=	char

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
model_thresh = load_model('model_bin.h5')
model_rgb = load_model('model_rgb_old.h5')

while camera.isOpened():
	ret, frame = camera.read()
	threshold = cv2.getTrackbarPos('trh1', 'trackbar')
	frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
	frame = cv2.flip(frame, 1)  # flip the frame horizontally
	cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 50),
					(frame.shape[1] , int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
	cv2.imshow('original', frame)

	count+=1


	###################
	# Keyboard OP
	k = cv2.waitKey(1)
	if k == 27:  # press ESC to exit
		break
	elif k == ord('b'):  # press 'b' to capture the background
		bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
		isBgCaptured = 1
		print('Background Captured')
	elif k == ord('r'):  # press 'r' to reset the background
		bgModel = None
		isBgCaptured = 0
		print ('Reset BackGround')

	###################

	#  Main operation
	cou = None
	img_cou = None
	aaa = None
	pred = None
	if isBgCaptured == 1:  # this part wont run until background captured
		img = removeBG(frame)
		img = img[50:int(cap_region_y_end * frame.shape[0]),
					int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
		cv2.imshow('mask', img)
		img_cou = copy.deepcopy(img)
	

		# convert t he image into binary image
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
		#cv2.imshow('blur', blur)
		ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
		#cv2.imshow('ori', thresh)



		# get the coutours
		thresh1 = copy.deepcopy(thresh)
		_,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		length = len(contours)
		maxArea = -1
		if length > 0:
			for i in range(length):  # find the biggest contour (according to area)
				temp = contours[i]
				area = cv2.contourArea(temp)
				if area > maxArea:
					maxArea = area
					ci = i

			cou = res = contours[ci]
			#cou = res
			aaa = np.reshape(cou,(1,-1,2))
			
		##########################################################################
		
		stencil = np.zeros(img_cou.shape).astype(img_cou.dtype)
		
		color = [255, 255, 255]
		cv2.fillPoly(stencil, aaa, color)
		result = cv2.bitwise_and(img_cou, stencil)
	
		#kernel = np.ones((3, 3), np.uint8)
		#result = cv2.erode(result, kernel, iterations=1)
		pred = medianfilter=cv2.medianBlur(result,3)
		cv2.imshow('cous', result)
		cv2.imshow('medianfilter ', medianfilter)


	if(isBgCaptured and (count % 50 == 0)):
		count = 1
		#print('!!!Save image!!!')
		img = cv2.flip(pred,1)#flip image		
		input_rgb = cv2.resize(img,(100,100)).reshape(1,100,100,3)
		out,prop=utils.predict(model_rgb,input_rgb)#predict
		if(mapCharacter(out)!='Nothing' and prop>70):
			#print('predict : ' ,mapCharacter(out), prop)
			#print('predict : ' ,mapCharacter(out))
			
			bow[index]=mapCharacter(out)
			check = True
			for i in range(2):
				 if(bow[i]!=mapCharacter(out)):
					 check = False
					 break
			if check:
				# print('sure : ' ,mapCharacter(out))
				sequence = mapCharacter(out)
				img_sequence = np.zeros((200,600,3), np.uint8)
				cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
				cv2.imshow('sequence', img_sequence)
				bow=['','','']
				

			index+=1
			if(index==2):
				index = 0
				


			
	

