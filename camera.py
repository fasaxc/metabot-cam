import time

import math
from cv2.cv import ShowImage
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

    # cv2.namedWindow("In", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("In", PREVIEW_RES[0], PREVIEW_RES[1])
    # cv2.namedWindow("Middle", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Middle", PREVIEW_RES[0], PREVIEW_RES[1])
    # cv2.namedWindow("Out", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Out", PREVIEW_RES[0], PREVIEW_RES[1])
    cv2.namedWindow("Out2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Out2", PREVIEW_RES[0], PREVIEW_RES[1])

    # capture frames from the camera
    last_time = time.time()
    for frame in cam.capture_continuous(rawCapture, format="yuv", use_video_port=True):
        # Get the Y component.
        image = frame.array[..., 0] # .copy()
        #cv2.imshow("In", image)
        #
        # big_blur = cv2.GaussianBlur(image, (101, 101), 0)
        # cv2.imshow("Middle", big_blur)
        # image = cv2.subtract(image, big_blur / 2)
        #
        # blur = cv2.GaussianBlur(image, (5, 5), 0)
        thr2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 101, 10)
        #cv2.imshow("Out", thr2)
        blurred_thr = cv2.GaussianBlur(thr2, (21, 21), 0)
        image2 = cv2.add(blurred_thr / 4, image)
        #cv2.imshow("Middle", image2)

        thr3 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 101, 10)
        # cv2.imshow("Out2", thr3)

        blurred = cv2.GaussianBlur(thr3, (31, 31), 0)

        key = cv2.waitKey(1) & 0xFF

        # Top left of image is (0, 0).
        darkest_x = None
        darkest_val = 256
        for x in xrange(W):
            v = blurred[H-5, x]
            if v < darkest_val:
                darkest_val = v
                darkest_x = x

        def get_pix(pix_x, pix_y):
            if 0 <= pix_x < W and 0 <= pix_y < H:
                return blurred[int(pix_y), int(pix_x)]

        def set_pix(pix_x, pix_y, value):
            if 0 <= pix_x < W and 0 <= pix_y < H:
                blurred[pix_y, pix_x] = value

        cur_x, cur_y = darkest_x, H-5
        prev_x, prev_y = darkest_x, H

        for i in xrange(40):
            set_pix(cur_x, cur_y, 180)

            d_x, d_y = cur_x - prev_x, cur_y - prev_y
            d_x_sq, d_y_sq = d_x ** 2, d_y ** 2
            norm = math.sqrt(d_x_sq + d_y_sq)
            d_x, d_y = d_x / norm, d_y / norm

            tap = 0
            darkest_test_x = None
            darkest_test_y = None
            darkest_val = 256
            for t_x, t_y in taps:
                rot_t_x = t_x * d_x - t_y * d_y
                rot_t_y = t_x * d_y + t_y * d_x

                test_x = cur_x + 10 * rot_t_x
                test_y = cur_y + 10 * rot_t_y

                val = get_pix(test_x, test_y)
                if val < darkest_val:
                    darkest_test_x = test_x
                    darkest_test_y = test_y
                    darkest_val = val

                tap += 1
            prev_x, prev_y = cur_x, cur_y
            for x in xrange(cur_x - 10, cur_x + 10):
                for y in xrange(cur_y - 10, cur_y + 10):
                    set_pix(x, y, 255)

            cur_x, cur_y = int(darkest_test_x), int(darkest_test_y)
            if not (0 <= cur_x < W and 0 <= cur_y < H):
                break

        now = time.time()
        frame_secs = now - last_time
        fps = 1 / frame_secs
        cv2.putText(blurred, "%.1f" % fps, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (20, 20, 20))
        last_time = now

        cv2.imshow("Out2", blurred)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
