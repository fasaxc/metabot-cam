import time

import math

import numpy
from cv2.cv import ShowImage, LogPolar, CV_WARP_FILL_OUTLIERS, fromarray
from picamera import PiCamera
from picamera.array import PiYUVArray
import time
import cv2
from scipy import signal

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
    cv2.namedWindow("Middle", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Middle", PREVIEW_RES[0], PREVIEW_RES[1])
    # cv2.namedWindow("Out", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Out", PREVIEW_RES[0], PREVIEW_RES[1])
    cv2.namedWindow("Out2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Out2", PREVIEW_RES[0], PREVIEW_RES[1])

    polar = cv2.cv.CreateMat(180, 180, cv2.cv.CV_8U)
    rotate90 = cv2.getRotationMatrix2D((90, 90), 90, 1)

    trace = numpy.zeros(shape=(180, 180), dtype=numpy.uint8)
    rot_integ = numpy.zeros(shape=(360,), dtype=numpy.uint8)

    window_fn = numpy.zeros(shape=(40,))
    for x in range(40):
        window_fn[x] = 1.0 - ((20.0 - x) * (20.0 - x) / 800.0)

    line_heading = 90
    output_heading = 90

    # capture frames from the camera
    last_time = time.time()
    for frame in cam.capture_continuous(rawCapture, format="yuv", use_video_port=True):
        # Get the Y component.
        image = frame.array[..., 0].copy()
        image2 = fromarray(image)

        LogPolar(image2, polar, (W/2, H/2), 50)
        image3 = numpy.asarray(polar)

        cv2.imshow("Middle", image3)

        # First figure out if we're on the line or not.  If we're on the line, we'll try
        # track in the right direction.  If we're off the line, we'll try to track towards it.
        # Sun the pixels out from the centre of the image.  If we're on the line, the near
        # pixels will be black.  The polar plot is log based so we see a very large blcak area.
        distance_integ = numpy.sum(image3, 0)
        min = numpy.min(distance_integ)
        max = numpy.max(distance_integ)
        distance_integ = (distance_integ - min) * 255 / (1 + max - min)
        distance_integ2 = distance_integ.astype(numpy.uint8)
        on_the_line = numpy.count_nonzero(numpy.less(distance_integ[:40], 127)) > 35

        # Now sum the plot in the other direction.  If we're on the line, this will give us
        # peaks in its direction.  Otherwise, there'll be one fuzzy peak in the general
        # direction of the line.
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

        if not on_the_line:
            min_idx = numpy.argmin(rot_integ)
            output_heading = min_idx
        else:
            window_min = line_heading - 20
            window_max = line_heading + 20
            if window_min < 0:
                window_min += 180
                window_max += 180
            elif window_max >= 360:
                window_min -= 180
                window_max -= 180
            window = rot_integ[window_min:window_max] * window_fn
            min_idx = numpy.argmin(window) + window_min
            if min_idx >= 180:
                line_heading = min_idx - 180
            else:
                line_heading = min_idx
            output_heading = line_heading

        trace[-2, line_heading] = 250
        trace[-2, output_heading] = 200

        now = time.time()
        frame_secs = now - last_time
        fps = 1 / frame_secs
        cv2.putText(image, "%.1f" % fps, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (20, 20, 20))
        last_time = now

        rads = 1.5 * math.pi - output_heading * 2 * math.pi / 180.0
        x = math.cos(rads) * 20
        y = math.sin(rads) * 20
        cv2.line(image, (W/2, H/2), (int(x + W/2), int(H/2 - y)), (200,200,200), 1)

        rads = 1.5 * math.pi - line_heading * 2 * math.pi / 180.0
        x = math.cos(rads) * 10
        y = math.sin(rads) * 10
        cv2.line(image, (W/2, H/2), (int(x + W/2), int(H/2 - y)), (250,250,250), 1)

        cv2.imshow("In", image)
        cv2.imshow("Out2", trace)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
