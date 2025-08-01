from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./chumma.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))
        # if detections_:
        #     track_ids = mot_tracker.update(np.asarray(detections_))
        # else:
        #     track_ids = []


        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test.csv')


# from ultralytics import YOLO
# import cv2
# import numpy as np
# import util
# from sort.sort import *
# from util import get_car, read_license_plate, write_csv

# # Result dictionary to hold outputs for all frames
# results = {}

# # Initialize SORT tracker
# mot_tracker = Sort()

# # Load YOLO models
# coco_model = YOLO('yolov8n.pt')  # For vehicle detection
# license_plate_detector = YOLO('./models/license_plate_detector.pt')  # For license plate detection

# # Load video
# cap = cv2.VideoCapture('./sample1.mp4')

# # COCO class IDs for vehicle types (bike to truck and more)
# vehicles = list(range(2, 10))  # IDs: bicycle, car, motorcycle, airplane, bus, train, truck, boat

# # Frame processing loop
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results[frame_nmr] = {}

#     # Detect vehicles
#     detections = coco_model(frame)[0]
#     detections_ = []
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])

#     # Handle empty detections to avoid SORT crashing
#     if len(detections_) > 0:
#         track_ids = mot_tracker.update(np.array(detections_))
#     else:
#         track_ids = []

#     # Detect license plates
#     license_plates = license_plate_detector(frame)[0]
#     for license_plate in license_plates.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = license_plate

#         # Match license plate to a vehicle
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#         if car_id != -1:
#             # Crop and process license plate
#             license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
#             license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

#             # OCR using EasyOCR
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

#             # Store results
#             results[frame_nmr][car_id] = {
#                 'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                 'license_plate': {
#                     'bbox': [x1, y1, x2, y2],
#                     'text': license_plate_text if license_plate_text else 'NA',
#                     'bbox_score': score,
#                     'text_score': license_plate_text_score if license_plate_text_score else 'NA'
#                 }
#             }

# # Write to CSV
# print("✅ Writing to test.csv...")
# write_csv(results, './test.csv')
# print("✅ Completed.")
