this project is for advanced lane find

1.Calibrate the camera:

    objp = np.zeros((6*9,3), np.float32)

    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space

    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/*.jpg')

    for idx, fname in enumerate(images):

    img = cv2.imread(fname)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[1], img.shape[0])
    # Find the chessboard corners
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)
     cv2.destroyAllWindows()
 I use 6x9 board .the image as follow:
![image]( https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/distort_image.png?raw=true)
![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/undistort_image.png?raw=true)


2.combine  color space and gradient thresholding to get the best of both worlds. the code as below:

    def img_threshold(img):
    

    :param img_: Input Image
    :return: Thresholded Image
    """
    distorted_img = np.copy(img)
    dst = cv2.undistort(distorted_img, mtx, dist, None, mtx)
    # Pull R
    R = dst[:,:,0]
    
    
    # Convert to HLS colorspace
    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS).astype(np.float)

    s_channel = hls[:,:,2]
    
    # Sobelx - takes the derivate in x, absolute value, then rescale
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize = 7)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= 10) & (scaled_sobelx <= 100)] = 1

    # Threshold R color channel
    R_binary = np.zeros_like(R)
    R_binary[(R >= 200) & (R <= 255)] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 125) & (s_channel <= 255)] = 1

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (sxbinary == 1)) | ((sxbinary == 1) & (R_binary == 1))
                     | ((s_binary == 1) & (R_binary == 1))] = 1
    return combined_binary.astype(np.uint8)
 threshold image as follow:
 ![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/Images/lane_image.png?raw=true)
 ![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/Images/threshold_image.png?raw=true)

 
 
3.Apply a perspective transform, choosing four source points manually,There are many other ways to select source points. For example, many perspective transform algorithms will programmatically detect four source points in an image based on edge or corner detection and analyzing attributes like color and surrounding pixels.next, we convert the four points on the original image to the transformed image .Note that the lanes in the transformed image are parallel.

def birds_eye(img, mtx, dist):

    binary_img = img_threshold(img)
    
    # Undistort
    undist = cv2.undistort(binary_img, mtx, dist, None, mtx)
    # Undistort
    
    # Grab the image shape
    img_size = (undist.shape[1], undist.shape[0])

    # Source points - defined area of lane line edges
    src = np.float32([[700,450],[1150,img_size[1]],[180,img_size[1]],[600,450]])

    # 4 destination points to transfer
    offset = 300 # offset for dst points
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])
    
    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Use cv2.warpPerspective() to warp the image to a top-down view
    top_down = cv2.warpPerspective(undist, M, img_size)

    return top_down, M
 ![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/Images/birds_eye%20image.png?raw=true)
 ![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/Images/warp%20image.png?raw=true)

4 Locate the Lane Lines and Fit a Polynomial
 implement sliding windows and fit a polynomial:First determine the approximate position of the left and right lane lines. This step is very simple. You only need to add the pixels in the picture along the y-axis to find the peaks around the middle point of the picture, that is, the possible area of the lane line, and then use it from the bottom up. Sliding window, calculate the non-zero pixel in the window. If the number of pixels is greater than a certain threshold, the average of these points is used as the center of the next sliding window.
 Fit a second order polynomial to each lane line:https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/096009a1-3d76-4290-92f3-055961019d5e/concepts/2f928913-21f6-4611-9055-01744acc344f
 ![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/color-fit-lines.jpg?raw=true)
 let's visualize the result here as well:
 ![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/Images/poly.image?raw=true)


5.draw the lane lines:
According above,the first Lines uses the birds eye image ,creates a histogram of where the binary activations occur,and uses sliding windows along the peak areas to estimate where the lane lines are.
because the lane lines are often lost,draw lines will first check whether the lines are detected.If not, go back up to first Lines.and resets to using new sliding windows below if upon failing five times in a row. If they are, we do not have to search the whole image for the lines. We can then draw the lines,as well as detect where the car is in relation to the middle of the lane,and what type of curvature it is driving at.
This part is not my algorithm, it comes from https://github.com/mvirgo/Advanced-Lane-Lines


6 Calculate the curvature of the lane line,according to some formula that come from https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/096009a1-3d76-4290-92f3-055961019d5e/concepts/2f928913-21f6-4611-9055-01744acc344f


7 Then use the perspective transformation to restore the fitted curve to the original perspective,and the last ,this process is implemented in each frame of the video to detect lane lines.

The result as follow:
![image](https://github.com/shiyangyang24/advance-lines-find-self-driving-p4/blob/master/Screenshot%20at%202018-04-04%2017:33:47.png?raw=true)

Disscussion


Because I am not very familiar with the lane detection algorithm,this solution works bad on challenge_video.mp4.When there is strong light and shadow, the detection of lane lines is very poor,even it may cause the lane line to be lost.Here's where I need to improve,algorithm improvement:when the camera does not detect a lane line, the program will die.What I need to improve is the rationalization of lane lines. When a lane line is detected, the other lane line is automatically found using the parallel line principle and the width of the lane line. When no lane line is detected, the last detected lane line is used.This will not cause the lane line to be lost.The second part I need to improve is the threshold value of the image. When encountering different scenarios, I need to choose the best threshold.








The second ,I use my camera to detect the lane lines in real secen.
we need apply another perspective transform,we select the different src and dst according to real secen.


Need to calibrate the camera and change the area of interest, because the camera is very bad in the first few frames, so we ignore the first few frames:

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    i=0
    while(cap.isOpened()):


     ret, frame = cap.read()
     i=i+1
     if ret ==True:
        if i>40:
            
            hit=process_video(frame)
            img_size = (frame.shape[1], frame.shape[0])
            
            cv2.imshow("kobe",hit)
            
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
       
    else: 
       break
    cap.release()
    cv2.destroyAllWindows()
    
    

















