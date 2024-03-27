from pywinauto import Desktop  # add this to handle UI scaling issue
import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
# from pyntcloud import PyntCloud
from threading import Thread
from test_delay_cal import delay_calculation
from test_system import *

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


def getMouseCoor(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y
        # print(mouseX,mouseY)


class cameraThread(Thread):

    def __init__(self, depth_image, depth_frame, depth_intrinsics,
                 depth_scale):
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
                point = rs.rs2_deproject_pixel_to_point(
                    self.depth_intrinsics, [i, j], depth)

                # Write the 3D coordinates to the file
                text = "%.5lf, %.5lf, %.5lf\n" % (point[0], point[1], point[2])
                f.write(text)

        # Close the file
        f.close()

    def run(self):
        global mouseX, mouseY
        #self.writeToFile()

        j = mouseY
        if j >= 1080:
            j -= 1080

        i = mouseX
        # depth = self.depth_frame.get_distance(i, j)


class delaysThread(Thread):

    def __init__(self, depthFrame, depthScale, depthMin, depthMax,
                 depthIntrinsics, colorIntrinsics, depthToColorExtrinsics,
                 colorToDepthExtrinsics):
        super(delaysThread, self).__init__()
        # self.hostIP     = '192.168.1.40'
        self.hostIP = '127.0.0.1'
        self.hostPort = 5004

        self.mode = 0
        self.micIndx = 0
        self.micGain = 0
        self.micDisable = 0

        self.setTest = 0
        self.micDelay = 0

        self.offsets = np.array([0, 0, 0])
        self.srcPos = np.array([0, 0, 0])

        self.depthFrame = depthFrame
        self.depthScale = depthScale
        self.depthMin = depthMin
        self.depthMax = depthMax
        self.depthIntrinsics = depthIntrinsics
        self.colorIntrinsics = colorIntrinsics
        self.depthToColorExtrinsics = depthToColorExtrinsics
        self.colorToDepthExtrinsics = colorToDepthExtrinsics

    def run(self):
        global logger  # logger disabled
        global mouseX, mouseY
        global point

        # clear lbl_info first
        sendBuf = b'SET0'

        # Deproject the pixel to a 3D point
        j = mouseY
        if j >= 1080:
            j -= 1080
        i = mouseX
        print("mouseX: %d  mouseY:%d" % (i, j))

        # project color pixel to depth pixel
        depthPixel = rs.rs2_project_color_pixel_to_depth_pixel(
            self.depthFrame.get_data(), self.depthScale, self.depthMin,
            self.depthMax, self.depthIntrinsics, self.colorIntrinsics,
            self.depthToColorExtrinsics, self.colorToDepthExtrinsics, [i, j])
        print("depthPixel: ", depthPixel)
        depth = self.depthFrame.get_distance(int(depthPixel[0]),
                                             int(depthPixel[1]))

        # project depth pixel to 3D point
        # x is right+, y is down+, z is forward+
        point = rs.rs2_deproject_pixel_to_point(
            self.depthIntrinsics,
            [int(depthPixel[0]), int(depthPixel[1])], depth)

        # initialize the position parameters
        # srcPos = point
        # self.srcPos = np.array([-1.0,-1.0,-1.0])
        self.srcPos = point
        self.offsets = [0, 0, 0]
        this_location = [6, 0.2, 0.3]

        # revised[add offsets],Brian,18 Mar 2024
        #Z=distance between camera and object, x is left+/right-, y is down+/up-
        delay = delay_calculation_v1(this_location)

        # logger.add_data('%s,%s' %('delay',np.array2string(delay)))
        #converting the delay into binary format
        delay_binary_output = delay_to_binary(delay)
        #print(delay_binary_output)
        #need to do later
        RW_field = [1, 1]
        mode = 0
        mic_gain = [1, 0]
        mic_num = 0
        en_bm = 1
        en_bc = 1
        mic_en = 1
        type = 0
        reserved = 0
        message = struct_packet(RW_field, mode, mic_gain, mic_num, en_bm,
                                en_bc, delay_binary_output[0], mic_en, type,
                                reserved)
        # print(message)
        messagehex = BintoINT(message)
        # print(messagehex)
        message1 = int(messagehex[2:4], 16)  # hex at  1 and 2
        message2 = int(messagehex[4:6], 16)  # hex at  3 and 4
        message3 = int(messagehex[6:8], 16)  # hex at  5 and 6
        message4 = int(messagehex[8:], 16)
        # print("m1:{},m2:{},m3:{},m4:{}\n".format(message1,message2,message3,message4))

        message5 = int(self.mode)  # mode
        message6 = int(self.micIndx)  # mic
        message7 = int(self.micGain)  # mic_gain
        message8 = int(self.micDisable)  # mic_disable
        message9 = int(self.setTest)  # set_test
        message10 = int(self.micDelay)  # mic_delay

        _, refDelay, _ = delay_calculation(self.srcPos, self.offsets[0],
                                           self.offsets[1], self.offsets[2])
        refDelay = refDelay * 48e3
        refDelay = np.max(refDelay) - refDelay
        refDelay = np.round(refDelay)

        #convert refDelay to byte
        #but make sure that they are within 0 to 255 first!!
        assert (refDelay >= 0).all() and (refDelay <= 255).all()

        refDelay = refDelay.astype(np.uint8)
        payload = refDelay.tobytes()
        # print('refDelay',refDelay)
        # print('payload',payload)
        # print('sendBuf',sendBuf)

        packet = prepareMicDelaysPacket(payload)
        if validateMicDelaysPacket(packet):
            print('packet ok')
        else:
            print('packet not ok')

        sendBuf = bytes([
            message1, message2, message3, message4, message5, message6,
            message7, message8, message9, message10
        ])

        # append packet to sendBuf
        sendBuf += packet

        # logger.add_data('data,%s,%s,%s' %(bytes(sendBuf),np.array2string(refDelay),np.array2string(self.srcPos)))

        # print("sendBuf: ",sendBuf)
        if send_and_receive_packet(self.hostIP,
                                   self.hostPort,
                                   sendBuf,
                                   timeout=3):
            print('data transmission ok')
            # self.showInfo('tx ok')
            # logger.add_data('tx ok')
        else:
            print('data transmission failed')
            # logger.add_data('tx failed')


if __name__ == '__main__':
    global mouseX, mouseY
    global point  # 3D coordinates

    mouseX = 0
    mouseY = 0
    prevI = 0
    prevJ = 0
    point = [0, 0, 0]

    DEPTH_CAM_WIDTH = 1280
    DEPTH_CAM_HEIGHT = 720

    COLOR_CAM_WIDTH = 1920
    COLOR_CAM_HEIGHT = 1080

    # initialize logger, disabled
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger(log_interval=1,
                        file_path="log/%s_sys.log" %
                        (timestamp))  # Specify the data file path
    logger.start_logging()

    # initialize the moving average calculator, window size =16 samples
    ma = MovingAverageCalculator(nSamples=16)

    # Create a pipeline object to manage streaming
    pipeline = rs.pipeline()

    # Create a config object to configure the streams
    config = rs.config()

    # point cloud
    pc = rs.pointcloud()

    # Enable the depth and color streams.
    config.enable_stream(rs.stream.depth, DEPTH_CAM_WIDTH, DEPTH_CAM_HEIGHT,
                         rs.format.z16, 30)
    config.enable_stream(rs.stream.color, COLOR_CAM_WIDTH, COLOR_CAM_HEIGHT,
                         rs.format.bgr8, 30)

    # Start the pipeline streaming
    profile = pipeline.start(config)

    # Create an align object to align the depth and color frames
    # align = rs.align(rs.stream.color)

    # Get the intrinsics of the color camera
    color_profile = rs.video_stream_profile(profile.get_stream(
        rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()

    # Get the intrinsics of the depth camera
    depth_profile = rs.video_stream_profile(profile.get_stream(
        rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    print("depth_intrinsics: ", depth_intrinsics)
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Get the depth scale of the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Get extrinsics
    depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)
    color_to_depth_extrinsics = color_profile.get_extrinsics_to(depth_profile)

    # for visualization
    depth_min = 0.1  #meter
    depth_max = 15.0  #meter

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset,
                         1)  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
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
        # aligned_frameset = align.process(frameset)

        # Get the aligned depth and color frames
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        # depth_frame = aligned_frameset.get_depth_frame()
        # color_frame = aligned_frameset.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert the images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # print("size: ", depth_image.size, color_image.size)
        j = mouseY
        if j >= 1080:
            j -= 1080

        i = mouseX

        # depth = depth_frame.get_distance(i, j)

        # Apply a color map to the depth image
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # depth_colormap = np.asanyarray(
        #     colorizer.colorize(depth_frame).get_data())

        # Stack the images vertically
        # images = np.vstack((color_image, depth_colormap))
        images = color_image

        # draw circle on images
        y1 = mouseY
        if y1 < 1080:
            y2 = y1 + 1080
        else:
            y2 = y1 - 1080

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.
        font_color = (0, 0, 255)  # White color
        line_type = cv2.LINE_AA

        # Write the debug message on the image

        text_size, _ = cv2.getTextSize("Debug Message", font, font_scale, 1)
        text_x = 10
        text_y = 10 + text_size[1]
        background_color = (50, 50, 50)  # dark grey background color

        # average_depth = ma.calculate_moving_average(depth)

        # targetPosDepthCam = 'dcam,%.2f,%.2f,%.2f' % (0, 0, average_depth)

        # Get the text size
        # (text_width, text_height), _ = cv2.getTextSize(targetPosDepthCam, font,
        #                                                font_scale, 1)
        # cv2.putText(images, targetPosDepthCam, (text_x, text_y), font,
        #             font_scale, font_color, 2, line_type)

        cv2.line(images,
                 (int(COLOR_CAM_WIDTH / 2 - 10), int(COLOR_CAM_HEIGHT / 2)),
                 (int(COLOR_CAM_WIDTH / 2 + 10), int(COLOR_CAM_HEIGHT / 2)),
                 (255, 0, 0), 3)

        cv2.line(images,
                 (int(COLOR_CAM_WIDTH / 2), int(COLOR_CAM_HEIGHT / 2 - 10)),
                 (int(COLOR_CAM_WIDTH / 2), int(COLOR_CAM_HEIGHT / 2 + 10)),
                 (255, 0, 0), 3)

        cv2.circle(images, (mouseX, y1), 3, (0, 0, 255), -1)
        cv2.circle(images, (mouseX, y2), 3, (0, 0, 255), -1)

        # add, Jason, 22 Mar 2024
        # get 3D coordinates and send delays
        if i != prevI and j != prevJ:
            prevI = i
            prevJ = j

            # initialize Delays
            sendDelays = delaysThread(depth_frame, depth_scale, depth_min,
                                      depth_max, depth_intrinsics,
                                      color_intrinsics,
                                      depth_to_color_extrinsics,
                                      color_to_depth_extrinsics)
            sendDelays.start()

        pointStr = 'x:%.2f,y:%.2f,z:%.2f' % (point[0], point[1], point[2])
        cv2.putText(images,
                    pointStr, (10, 62),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        # Show the images
        cv2.imshow('Realsense', images)
        cv2.setMouseCallback('Realsense', getMouseCoor)

        cur_time = datetime.datetime.now()

        if cur_time - prev_time > datetime.timedelta(seconds=5):
            prev_time = cur_time
            #print("save file")
            t = cameraThread(depth_image, depth_frame, depth_intrinsics,
                             depth_scale)
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
    logger.stop_logging()
    cv2.destroyAllWindows()
