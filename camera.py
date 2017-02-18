import time

from cv2.cv import ShowImage
from picamera import PiCamera
from picamera.array import PiYUVArray
import time
import cv2

CAM_RES = (128, 80)
PREVIEW_RES = tuple(v * 4 for v in CAM_RES)

with PiCamera() as cam:
    cam.resolution = CAM_RES
    cam.rotation = 180
    cam.framerate = 50

    rawCapture = PiYUVArray(cam, size=CAM_RES)

    # allow the camera to warmup
    time.sleep(0.1)

    cv2.namedWindow("In", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("In", PREVIEW_RES[0], PREVIEW_RES[1])
    cv2.namedWindow("Out", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Out", PREVIEW_RES[0], PREVIEW_RES[1])

    # capture frames from the camera
    for frame in cam.capture_continuous(rawCapture, format="yuv", use_video_port=True):
        # Get the Y component.
        image = frame.array[..., 0] # .copy()
        cv2.imshow("In", image)

        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # show the frame
        cv2.imshow("Out", image)
        key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
