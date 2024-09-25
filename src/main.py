import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import google.auth
from google.cloud import vision
from google.cloud.vision_v1 import types
import json
import re
from bounds_ui import select_lines_ui, display_selections
from ocr import google_ocr, annotations_to_lines, calculate_line_height_threshold, preprocess_annotations, estimate_font_size



if __name__ == "__main__":
    # pages = convert_from_path('mksa.pdf', dpi=400)

    # # Save the image as a PNG file
    # first_page.save("output_mksa.png", "PNG")
    img = cv2.imread('../output_mksa.png')

    selected_lines = select_lines_ui(img)

    display_selections(img, selected_lines)

    body = img[selected_lines[0]:selected_lines[1], :]
    footnotes = img[selected_lines[1]:selected_lines[2], :]


    body_annotations = google_ocr(body)

    word_list, bounding_polys = preprocess_annotations(body_annotations)
    line_height_threshold = calculate_line_height_threshold(bounding_polys)
    lines = annotations_to_lines(word_list, line_height_threshold)
    reordered_text = '\n'.join([' '.join(line) for line in lines])

    avg_height, height_std = estimate_font_size(word_list, line_height_threshold)
    print("avg height: ", avg_height)
    print("height std: ", height_std)


    with open('output_mksa.txt', 'w', encoding='utf-8') as f:
        f.write(reordered_text)
    print("wrote text to output_mksa.txt")
    # TODO:
    # find height, std of one page

    # send full images to google
    # isolate bodies and concat
    # save

    # words = ["مرحبا 123", "١٢٣ ( 456 )", "( hello )", "سلام"]
    # results = {word: is_valid_arabic_word(word) for word in words}
    # print(results)

    # Calculate the average height of the contours (which gives an idea of the font size)

    # with open('output_mksa.txt', 'w', encoding='utf-8') as f:
    #     f.write(text)
