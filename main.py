###
# YT: Computer vision engineer
#https://www.youtube.com/watch?v=F-884J2mnOY
###

import cv2
import os
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not

#### DATA PATHS
BASE_PATH = '../../data/parking_counter'

VIDEO_PATH = os.path.join(BASE_PATH, 'parking_1920_1080_loop.mp4')
MASK_PATH = os.path.join(BASE_PATH, 'mask_1920_1080.png')


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

### Load files
mask = cv2.imread(MASK_PATH, 0) #0: open as grayscale
cap = cv2.VideoCapture(VIDEO_PATH)

### Bounding boxes
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)


spots_status = [None] * len(spots)
diffs = [None] * len(spots) #Measure if there is something going on


previous_frame = None
step = 30 #How often (in frames) to run classification of spot
frame_num = 0

ret = True

while ret:
    ret, frame = cap.read()

    if frame_num % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]

            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w, :])

    if frame_num % step == 0:
        if previous_frame is None:
            arr = range(len(spots))
        else:
            arr = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs) > 0.4]
        ### print bboxes
        for spot_idx in arr:
            spot = spots[spot_idx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            spot_status = empty_or_not(spot_crop)

            spots_status[spot_idx] = spot_status

        previous_frame = frame.copy()
        

    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spot

        color_vector = (0,255,0) if spot_status == True else (0,0,255)
        frame = cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color_vector, 2)

    cv2.rectangle(frame, (80,20), (550,80), (0,0,0), -1)
    cv2.putText(frame, 'Available spots: {}/{}'.format(sum(spots_status), len(spots_status)), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()