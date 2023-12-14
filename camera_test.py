import os
import cv2
import cv2 as cv
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model = model_2


model_test = AntiSpoofPredict(0)
# Create a VideoCapture object called cap
cap = cv.VideoCapture(0)

# This is an infinite loop that will continue to run until the user presses the `q` key
while cap.isOpened():
    tic = time.time()

    # Read a frame from the webcam
    ret, frame_root = cap.read()
    frame = frame_root.copy()
    # If the frame was not successfully captured, break out of the loop
    if ret is False:
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tic = time.time()
    image_bbox = model_test.get_bbox(frame)
    # print('Detection time: ', (time.time() - tic))
    prediction = np.zeros((1, 3))
    
    for model in [model_1, model_2]:
        model_name = model.split('/')[-1]
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img, img_ = CropImage().crop(**param)
        cv.imshow("face", img_)
        # print(np.shape(img))
        # start = time.time()
        prediction += model_test.predict(img, model)
        
    test_speed = time.time() - tic
    fps = 1/test_speed

    label = np.argmax(prediction)
    value = prediction[0][label]

    if label == 1:
        # print("Image is Real Face. Score: {:.2f}.".format( value))
        result_text = "RealFace Score: {}".format(value)
        color = (0, 255, 0)
    else:
        # print("Image is Fake Face. Score: {:.2f}.".format( value))
        result_text = "FakeFace Score: {}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {} s".format(test_speed))
    cv2.rectangle(
        frame_root,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        frame_root,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)

    # Display the FPS on the frame
    cv.putText(frame_root, f"FPS: {fps}", (30, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv.LINE_AA)

    # Display the frame on the screen
    cv.imshow("frame", frame_root)

    # Check if the user has pressed the `q` key, if yes then close the program.
    key = cv.waitKey(1)
    if key == ord("q"):
        break

# Release the VideoCapture object
cap.release()

# Close all open windows
cv.destroyAllWindows()