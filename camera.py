import time

import math

import numpy
from cv2.cv import ShowImage, LogPolar, CV_WARP_FILL_OUTLIERS, fromarray
from picamera import PiCamera
from picamera.array import PiYUVArray
import time
import cv2

CAM_RES = (128, 80)
W = CAM_RES[0]
H = CAM_RES[1]
PREVIEW_RES = tuple(v * 2 for v in CAM_RES)

taps = [(math.cos((i-4) * math.pi/8), math.sin((i-4) * math.pi/8)) for i in range(9)]

with PiCamera() as cam:
    cam.resolution = CAM_RES
    cam.rotation = 180
    cam.framerate = 50

    rawCapture = PiYUVArray(cam, size=CAM_RES)

    # allow the camera to warmup
    time.sleep(0.1)

    cv2.namedWindow("In", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("In", PREVIEW_RES[0], PREVIEW_RES[1])
    # cv2.namedWindow("Middle", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Middle", PREVIEW_RES[0], PREVIEW_RES[1])
    # cv2.namedWindow("Out", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Out", PREVIEW_RES[0], PREVIEW_RES[1])
    cv2.namedWindow("Out2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Out2", PREVIEW_RES[0], PREVIEW_RES[1])

    polar = cv2.cv.CreateMat(180, 180, cv2.cv.CV_8U)
    rotate90 = cv2.getRotationMatrix2D((90, 90), 90, 1)

    trace = numpy.zeros(shape=(180, 180), dtype=numpy.uint8)
    rot_integ = numpy.zeros(shape=(360,), dtype=numpy.uint8)

    window_fn = numpy.arange(1.0, 41) / 40.0

    heading = None

    # capture frames from the camera
    last_time = time.time()
    for frame in cam.capture_continuous(rawCapture, format="yuv", use_video_port=True):
        # Get the Y component.
        image = frame.array[..., 0].copy()
        image2 = fromarray(image)

        LogPolar(image2, polar, (W/2, H/2), 50)
        image3 = numpy.asarray(polar)

        integ = numpy.sum(image3, 1)
        min = numpy.min(integ)
        max = numpy.max(integ)
        integ = (integ - min) * 255 / (1 + max - min)
        integ2 = integ.astype(numpy.uint8)

        rot_integ[:135] = integ2[45:]
        rot_integ[135:180] = integ2[:45]
        rot_integ[180:] = rot_integ[:180]

        trace[:-1, :] = trace[1:,:]
        trace[-1, :] = rot_integ[:180]

        if heading is None:
            # haven't locked onto line yet
            min_idx = numpy.argmin(rot_integ)
            heading = min_idx
        else:
            window_min = heading - 20
            window_max = heading + 20
            if window_min < 0:
                window_min += 180
                window_max += 180
            elif window_max >= 360:
                window_min -= 180
                window_max -= 180
            window = rot_integ[window_min:window_max] * window_fn
            min_idx = numpy.argmin(window) + window_min
            if min_idx >= 180:
                heading = min_idx - 180
            else:
                heading = min_idx

        trace[-2,heading] = 200
        print heading

        now = time.time()
        frame_secs = now - last_time
        fps = 1 / frame_secs
        cv2.putText(image, "%.1f" % fps, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (20, 20, 20))
        last_time = now

        rads = 1.5 * math.pi - heading * 2 * math.pi / 180.0
        x = math.cos(rads) * 20
        y = math.sin(rads) * 20

        cv2.line(image, (W/2, H/2), (int(x + W/2), int(H/2 - y)), (200,200,200), 1)

        cv2.imshow("In", image)
        cv2.imshow("Out2", trace)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
