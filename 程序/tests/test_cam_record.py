import cv2
from pywinauto import Desktop

# Open the video capture device (e.g., webcam)
cap = cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

# Loop to capture frames and write to the output file
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Write the frame to the output file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()