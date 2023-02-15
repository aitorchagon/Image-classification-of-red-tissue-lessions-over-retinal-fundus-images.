import cv2
import numpy as np
import math




# Blob detection parameters (for notch detection). We define a blob as 
#a region where there is some inhomogeneity in brightness or color.
BLOB_PARAMS = cv2.SimpleBlobDetector_Params()
BLOB_PARAMS.minThreshold = 0.0
BLOB_PARAMS.maxThreshold = 30
BLOB_PARAMS.thresholdStep = 15
BLOB_PARAMS.filterByArea = False
BLOB_PARAMS.filterByColor = False
BLOB_PARAMS.filterByConvexity = False
BLOB_PARAMS.filterByInertia = True
BLOB_PARAMS.minInertiaRatio = 0.05
BLOB_PARAMS.maxInertiaRatio = 1

def clahe_algorithm(image):
    '''
    This algorithm of equalization is based on Clahe, which operates on small regions.
    It is normally very useful when applied to a single channel; 
    in our case, the green channel of the RGB colour space.

    Parameters
    ----------
    image : array
        Our image to be equalized.

    Returns
    -------
    final_img : array
        Our image after being equalized.

    '''
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    final_img = clahe.apply(image)
    return final_img

def preprocess_image(path):
    '''
    Loads an image, converts to grayscale, constructs our bounding box so 
    we can place a square over our notch in general .
    Afterwards, we will flip the image if necessary based
    on which eye are we studying on the image and whether we have found a notch.
    Finally, we will equalize our image using CLAHE algorithm. 

    Parameters
    ----------
    path :string
        The path where we are going to analyze (path of our files)

    Returns
    -------
    output : array
        Our image being preprocessed.

    '''
    # Loading the image also converts it to grayscale.
    img = cv2.imread(path, 0)
    _, thresholding = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    #img = bb_resize(img, thresholding)
    _, img_thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Two-part notch-detection. Notch could be either in upper-right quadrant or the bottom-right quadrant. 
    #We will start with the first one. If we do not detect any notch, we won't do any flipping.
    if detect_notch(img, img_thresh):
        pass
        #cv2.flip(img, -1, img)

    else:
        pass
        vert_flip = cv2.flip(img, 0)
        vert_flip_thresh = cv2.flip(img_thresh, 0)

        if detect_notch(vert_flip, vert_flip_thresh):
            pass
            cv2.flip(img, -1, img)

    # Examine the file name and flip the eye horizontally if it's a left eye.
    if "left" in path:
        pass
        cv2.flip(img, 1, img)

    # Finally, equalize the image.
    output = clahe_algorithm(img)

    return output


def load_image(path):
    """
    Loads an image, transforms it to grayscale, and resizes it. 
    """

    img_in = cv2.imread(path, 0)
    output = cv2.threshold(img_in, 30, 255, cv2.THRESH_BINARY)
    img_in = bb_resize(img_in, output)

    return output


def hough_circles(img):
    '''
    Apply Hough Circle Transform. If no circles are found, we return an empty list.

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''
    img1 = img.copy()

    # Forcing Hough transform to align to our notch edge, reducing the edges left over
    #after our subtraction. 
    height, width = img1.shape
    cv2.rectangle(img, (0, 0), (int(width / 2), height),
                 (0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(img, (0, height), (width, int(height / 2)),
                 (0, 0, 0), thickness=cv2.FILLED)

    circles = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 2, 512,
                              param1=140, param2=30,
                              minRadius=204, maxRadius=512)

    output = []
    if circles is not None:
        circles = circles[0]
        for c in circles[0:]:
            output.append((c[0], c[1], c[2]))
            
    
    return output


def bb_resize(img, img_thresh):
    """
    Resizes an image using bounding boxes. This is done by thresholding the
    image and then calculating its bounding box. The shorter dimension of the
    bounding box is then expanded (with black pixels) to make the bounding box
    square. The pixels in the bounding box are moved to a new image, which is
    then resized to a standard resolution.

    This effect of this process should be that any eyeball image is roughly
    centered at the same position and about the same size. This is important for
    notch detection so that a small square can be placed approximately over
    where the notch should be in a standardized image.
    """
    x, y, w, h = cv2.boundingRect(img_thresh)

    # If no bounding rectangle was able to be formed, then the image is
    # probably completely unusable. Simply resize to standard resolution and
    # move on.
    if (w == 0) or (h == 0):
        return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Destination canvas is square, length of max dimension of the bb.
    max_wh = max(w, h)
    img_expand = np.zeros((max_wh, max_wh), dtype=np.uint8)

    # Copy the bounding box (region of interest) into the center of the
    # destination canvas. This will produce black bars on the short side.
    diff_w = max_wh - w
    diff_h = max_wh - h
    half_dw = int(math.floor(diff_w / 2.0))
    half_dh = int(math.floor(diff_h / 2.0))
    roi = img[y:y + h, x:x + w]
    img_expand[half_dh:(half_dh + h), half_dw:(half_dw + w)] = roi

    # Resize to our standard resolution.
    return cv2.resize(img_expand, (256, 256), interpolation=cv2.INTER_AREA)


def detect_notch(img, img_thresh):
    """
    Detects if a notch is present in the image, and if so, returns True.

    First, the Hough Circle Transform is applied to find a circle corresponding
    to our FOV (field of view), obtained by a thresholded image. Ideally we should only
    be having either our notch or nothing. Since we will likely have some artifacts left from the
    edges of the subtraction, we will open (morphological opening) our image to avoid that.

    A region of interest (ROI) is positioned over where we usually find our notches,
    which is in the NE quadrant (45 degrees, indeed).
    Blob detection is run over this ROI. If a blob is detected, it is assumed
    that the blob is a notch, and the function can return True.
    """
    circles = hough_circles(img)

    # Paint out the FOV.
    if circles:
        x, y, r = circles[0]
        cv2.circle(img_thresh, (int(x), int(y)), int(r), (0, 0, 0), cv2.FILLED)
    else:
        return False


    # Opening the image to take care of our artifacts.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # Extract a region of interest that is very likely to contain the notch if
    # one is present in the image. This corresponds to a small square at about
    # the 45 degree mark on the eyeball. Some notches are slightly lower than
    # this, so the ROI should be large enough to capture many notch positions.
    roi_size = img_thresh.shape[0] * 0.25
    half_rs = 0.125

    # Finding our cartesian coordinates for our 45 degrees point at the NE.
    radius = img_thresh.shape[0] / 2.0
    angle = math.pi / 4.0
    side = radius * math.sin(angle)

    # Getting the ROI
    x, y = (int(radius + side - half_rs), int(radius - side - half_rs))
    roi = img_thresh[y:int(y + roi_size), x:int(x + roi_size)]

    # Run blob detection on what's left.
    sbd = cv2.SimpleBlobDetector_create(BLOB_PARAMS)
    keypoints = sbd.detect(roi)

    # If keypoints were found, then we assume that a notch was detected.
    return bool(keypoints)
