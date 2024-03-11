from pywinauto import Desktop  # add this to handle UI scaling issue
import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
# from pyntcloud import PyntCloud
from threading import Thread

# pip install pyrealsense2
# pip install pyntcloud


# Add[for getting moving averages],Brian,29 Feb 2024
class MovingAverageCalculator:
    def __init__(self, nSamples):
        self.nSamples = nSamples
        self.window = []
    
    def calculate_moving_average(self, x):
        self.window.append(x)

        if len(self.window) > self.nSamples:
            self.window.pop(0)

        average = sum(self.window) / len(self.window)
        return average


def getMouseCoor(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y
        print(mouseX,mouseY)

class cameraThread(Thread):
    def __init__(self, depth_image, depth_frame, depth_intrinsics, depth_scale):
        super(cameraThread, self).__init__()
        self.depth_image = depth_image
        self.depth_frame = depth_frame
        self.depth_intrinsics = depth_intrinsics
        self.depth_scale = depth_scale

    def writeToFile(self):
        # Create a file to write the 3d coordinates
        f = open("3d_coordinates.txt", "a")
        f.write(str(datetime.datetime.now()) + '\n')

        # Loop over all the pixels in the depth image
        rows, cols = self.depth_image.shape
        
        for i in range(rows):
            for j in range(cols):
                # Get the depth value at the pixel
                depth = self.depth_frame.get_distance(j, i)

                # Deproject the pixel to a 3D point
                point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [j, i], depth * self.depth_scale)

                # Write the 3D coordinates to the file
                text = "%.5lf, %.5lf, %.5lf\n" % (point[0], point[1], point[2])
                f.write(text)
        
        # Close the file
        f.close()

    def run(self):
        global mouseX,mouseY
        #self.writeToFile()

        j = mouseY
        if j>=1080:
            j-=1080

        i = mouseX
        depth = self.depth_frame.get_distance(i,j)
        

        # Deproject the pixel to a 3D point
        point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [j,i], depth * self.depth_scale)
        #print(depth,point)
        
        
    
if __name__=='__main__':
    global mouseX, mouseY

    mouseX = 0
    mouseY = 0

    DEPTH_CAM_WIDTH  = 1280
    DEPTH_CAM_HEIGHT = 720

    COLOR_CAM_WIDTH  = 1920
    COLOR_CAM_HEIGHT = 1080

    # initialize the moving average calculator, window size =16 samples
    ma = MovingAverageCalculator(nSamples=16)

    # Create a pipeline object to manage streaming
    pipeline = rs.pipeline()

    # Create a config object to configure the streams
    config = rs.config()

    # point cloud
    pc = rs.pointcloud()

    # Enable the depth and color streams
    config.enable_stream(rs.stream.depth, DEPTH_CAM_WIDTH, DEPTH_CAM_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, COLOR_CAM_WIDTH, COLOR_CAM_HEIGHT, rs.format.bgr8, 30)

    # Start the pipeline streaming
    profile = pipeline.start(config)

    # Create an align object to align the depth and color frames
    align = rs.align(rs.stream.color)

    # Get the intrinsics of the color camera
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()

    # Get the intrinsics of the depth camera
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    print("depth_intrinsics: ", depth_intrinsics)
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Get the depth scale of the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    # for visualization
    depth_min = 0.1 #meter
    depth_max = 15.0 #meter

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
    colorizer.set_option(rs.option.min_distance, depth_min)
    colorizer.set_option(rs.option.max_distance, depth_max)

    # Create a window to display the images
    cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)

    prev_time = datetime.datetime.now()
    # Loop until the user presses ESC key
    while True:
        

        # Get a frameset from the pipeline
        frameset = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frameset = align.process(frameset)

        # Get the aligned depth and color frames
        depth_frame = aligned_frameset.get_depth_frame()
        color_frame = aligned_frameset.get_color_frame()


        
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert the images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        j = mouseY
        if j>=1080:
            j-=1080

        i = mouseX

        depth = depth_frame.get_distance(i,j)

        # Apply a color map to the depth image
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Stack the images horizontally
        images = np.vstack((color_image, depth_colormap))

        # draw circle on images
        y1 = mouseY
        if y1<1080:
            y2 = y1+1080
        else:
            y2 = y1-1080


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.
        font_color = (0, 0, 255)  # White color
        line_type = cv2.LINE_AA

        # Write the debug message on the image
        
        text_size, _ = cv2.getTextSize("Debug Message", font, font_scale, 1)
        text_x = 10
        text_y = 10 + text_size[1]
        background_color = (50, 50, 50)  # dark grey background color


        average_depth = ma.calculate_moving_average(depth)

        targetPosDepthCam = 'dcam,%.2f,%.2f,%.2f' %(0,0,average_depth)

        # Get the text size
        (text_width, text_height), _ = cv2.getTextSize(targetPosDepthCam, font, font_scale, 1)
        
        cv2.putText(images, targetPosDepthCam, (text_x, text_y), font, font_scale, font_color, 2, line_type)
        cv2.circle(images,(mouseX,y1),3,(0,0,255),-1)
        cv2.circle(images,(mouseX,y2),3,(0,0,255),-1)


        # Show the images
        cv2.imshow('Realsense', images)
        cv2.setMouseCallback('Realsense',getMouseCoor)



        # point cloud
        # points = pc.calculate(depth_frame)
        # points.export_to_ply("1.ply", color_frame)
        # cloud = PyntCloud.from_file("1.ply")
        # cloud.plot()

        # vtx = np.asanyarray(points.get_vertices())
        # tex = np.asanyarray(points.get_texture_coordinates())
        # print(type(points), points)
        # print(type(vtx), vtx.shape, vtx)
        # print(type(tex), tex.shape, tex)    

        cur_time = datetime.datetime.now()

        if cur_time - prev_time > datetime.timedelta(seconds=5):
            prev_time = cur_time
            #print("save file")
            t = cameraThread(depth_image, depth_frame, depth_intrinsics, depth_scale)
            t.start()

        # Check if the user pressed ESC key
        key = cv2.waitKey(1)
        if key == 27:
            break

        # If the window is closed, break the loop
        prop = cv2.getWindowProperty("Realsense", cv2.WND_PROP_VISIBLE)
        
        if prop == 0:
            break







    # Stop the pipeline and close the window
    pipeline.stop()
    cv2.destroyAllWindows()