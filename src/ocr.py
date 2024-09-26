import cv2
import numpy as np
import google.auth
from google.cloud import vision
from google.cloud.vision_v1 import types
import re
import os

credentials, project = google.auth.default()
client = vision.ImageAnnotatorClient(credentials=credentials)

def google_ocr(img):
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        print("Error: Could not encode the image.")
        return

    # Prepare image for Google Vision API
    content = encoded_image.tobytes()
    image_for_vision = types.Image(content=content)

    # Add a language hint for Arabic ("ar")
    image_context = vision.ImageContext(language_hints=['ar'])

    response = client.document_text_detection(image=image_for_vision, image_context=image_context)
    if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")
    return response.text_annotations

def find_footnote_line(img, min_y, i):
    im_h, im_w, im_d = img.shape
    print("img shape: ", img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,10))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite(f"temp/output_{i}_processed.png", dilate)
    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h < 40 and w > 600 and x > im_w/4 and y > min_y:
            print("found rect: ", x, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
            return y
    return im_h
    # cv2.imwrite("output_footnote.png", roi)
