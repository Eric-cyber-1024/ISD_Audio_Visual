''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import numpy as np

def calculate_distance(focal_length, known_width, pixel_width):
    distance = (known_width * focal_length) / pixel_width
    return distance

# Example values
focal_length = 60  # Focal length of the camera in pixels (you need to calibrate or obtain this value)
known_width = 10     # Width of the known object in real-world units (e.g., centimeters or inches)
pixel_width = 50     # Pixel width of the object in the image

# Calculate distance using the formula
distance = calculate_distance(focal_length, known_width, pixel_width)

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
# haarcascade = "haarcascade_righteye_2splits_opencv4.xml"
haarcascade_clf = "data/" + haarcascade
# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)
# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
# LBFmodel = "LBF686_GTX.yaml"
LBFmodel_file = "data/" + LBFmodel



# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)
# get image from webcam
webcam_cap = cv2.VideoCapture(2)
count = 0 

standard_face_width = 0.14 # cm 

while(True):
    # read webcam
    _, frame = webcam_cap.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(gray)
    # faces = faces[0]

    for (x,y,w,d) in faces:
        # Detect landmarks on "gray"
        _, landmarks = landmark_detector.fit(gray, np.array(faces))
        
        for landmark in landmarks:
            
            # for x,y in landmark[0]:
                
            #     # display landmarks on "frame/image,"
            #     # with blue colour in BGR and thickness 2
            x1,y1 = landmark[0][0] 
            x2,y2 = landmark[0][16]
            cv2.circle(frame, (int(x1), int(y1)), 1, (255, 0, 0), 2)
            cv2.circle(frame, (int(x2), int(y2)), 1, (255, 0, 0), 2)
            
            pixel_distance = np.sqrt(((abs(int(y1-y2)))^2 +(abs(int(x1-x2)))^2))

            distance = calculate_distance(focal_length, standard_face_width, pixel_distance)

            format_distance = "{:.2f}".format(distance)
            cv2.putText(frame,format_distance,(0,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,200,100),2,cv2.LINE_AA)
            # print(distance ,int(y1-y2) ,  int(x1-x2))
                
                # count+=1
    # print(count)
    count = 1


    # save last instance of detected image
    # cv2.imwrite('face-detect.jpg', frame)    
    
    # Show image
    cv2.imshow("frame", frame)

    # terminate the capture window
    if cv2.waitKey(20) & 0xFF  == ord('q'):
        webcam_cap.release()
        cv2.destroyAllWindows()
        break