import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessed import clahe_algorithm

def extract_bv(image):	
    '''
    This function allows us to extract our blood vessels. To do that,
    we will split our image in three channels and work with the green one. Afterwards,
    we will apply ASF (alterante sequential filtering), based on opening/closing operations
    with varying Structuring Elements (each time the SE being larger).
    Finally, we use snakes (a method to draw contours) to extract our blood vessels and some noise.

    Parameters
    ----------
    image : array
        Our input image.

    Returns
    -------
    blood_vessels: array
        Our image with the blood vessels of the fundus.

    '''	
    blue,green,red = cv2.split(image)
    green = clahe_algorithm(green)

	# applying alternate sequential filtering (3 times closing opening)
	
    step1 = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    step2 = cv2.morphologyEx(step1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    step3 = cv2.morphologyEx(step2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    step4 = cv2.morphologyEx(step3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)	
    step5 = cv2.morphologyEx(step4, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23, 23)), iterations = 1)	
    step6 = cv2.morphologyEx(step5, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23, 23)), iterations = 1)		
    step7 = cv2.subtract(step6,green)	
    step8 = clahe_algorithm(step7)		

	# noise removal (very small contours) using snakes algorithm.	
    ret,step9 = cv2.threshold(step8,15,255,cv2.THRESH_BINARY)		
    mask = np.ones(step8.shape[:2], dtype="uint8") * 255		
    contours_1, kk = cv2.findContours(step9.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	
    for contour in contours_1:		
        if cv2.contourArea(contour) <= 200:			
            cv2.drawContours(mask, [contour], -1, 0, -1)			
	
    im = cv2.bitwise_and(step8, step8, mask=mask)	
    _,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)				
    fin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

	# removing blood vessels and getting them on another image
	
    fundus_eroded = cv2.bitwise_not(fin)		
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255	
    contours_2, k = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	
    for contour in contours_2:
		
        shape = "None"	
        perimeter = cv2.arcLength(contour, True)		
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, False)   				
		
        if len(approx) > 4 and cv2.contourArea(contour) <= 3000 and cv2.contourArea(contour) >= 100:			
            shape = "circle"	
		
        else:			
            shape = "veins"		
        if(shape=="circle"):			
            cv2.drawContours(xmask, [contour], -1, 0, -1)	

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels



#img = cv2.imread('/home/aitorchagon/Desktop/AIM/proyecto/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/image060.png')
#vessels=extract_bv(img)
#figure = plt.figure(figsize=(20,10))
#plt.subplot(121)
#plt.imshow(img)
#plt.title("Image")
#plt.subplot(122)
#plt.imshow(vessels)
#plt.title("Vessels")

	
