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

def preprocess_annotations(annotations):
    word_list = []
    bounding_polys = []

    # Iterate through each word (skip the first annotation since it includes the whole text)
    for annotation in annotations[1:]:
        # Extract the vertices of the bounding polygon
        vertices = annotation.bounding_poly.vertices
        bounding_polys.append(annotation.bounding_poly)
        # Get the x-coordinate of the top-left vertex (for ordering horizontally)
        top_left_x = vertices[0].x
        top_left_y = vertices[0].y
        top_y = min([vertex.y for vertex in vertices])
        bottom_y = max([vertex.y for vertex in vertices])

        # Calculate the height of the bounding box
        height = abs(bottom_y - top_y)
        # Add the word and its position to the list
        word_list.append((annotation.description, top_left_x, top_left_y, height))
    # Sort the words: first by y-coordinate (vertical), then by x-coordinate (horizontal)
    word_list.sort(key=lambda w: (w[2], -w[1]))
    return word_list, bounding_polys

def annotations_to_lines(word_list, line_height_threshold):
    # Create an empty list to store the reordered lines
    lines = []
    current_line = []
    current_y = word_list[0][2] if word_list else 0

    for word, x, y, h in word_list:
        if abs(y - current_y) > line_height_threshold:
            lines.append(current_line)
            current_line = []
            current_y = y
        current_line.append(word)

    # Append the last line
    if current_line:
        lines.append(current_line)

    return lines

def calculate_line_height_threshold(bounding_polys):

    """
        Given a list of bounding boxes, dynamically calculates a line height threshold
        to detect new lines using the top-right vertex of each bounding box without specifying a tolerance.

        Parameters:
            bounding_boxes (list of tuples): A list where each element is a bounding box.
                                            Each bounding box is a tuple in the form:
                                            (x_left, y_top, x_right, y_bottom).

        Returns:
            float: The dynamically determined line height threshold.
    """

    # Extract the y_top values from each bounding box (we only need y_top for the top-right vertex)
    y_top_values = [poly.vertices[3].y for poly in bounding_polys]

    # Sort the y_top values in ascending order
    y_top_values.sort()

    # Calculate the vertical distances between adjacent top-right y values
    vertical_distances = [y_top_values[i+1] - y_top_values[i] for i in range(len(y_top_values) - 1)]

    # If there are no distances, return 0
    if len(vertical_distances) == 0:
        return 0

    # Dynamically detect significant gaps using a statistical method (e.g., z-score)
    mean_distance = np.mean(vertical_distances)
    std_distance = np.std(vertical_distances)

    # Define a dynamic threshold based on significant deviations from the mean
    # This will help identify larger gaps that signify a new line
    significant_gaps = [dist for dist in vertical_distances if dist > (mean_distance + 2 * std_distance)]

    # If we find significant gaps, we assume they represent the distances between lines
    if significant_gaps:
        return np.median(significant_gaps)
    else:
        # If no significant gaps are found, use the median of all distances
        return np.median(vertical_distances)


def is_valid_arabic_word(word):
    # Define the regex pattern, allowing whitespace characters
    pattern = r'^[\u0600-\u06FF\sA-Za-z$€£]*$'

    # Use re.fullmatch to check if the entire word matches the pattern
    return True if re.fullmatch(pattern, word) else False


def estimate_font_size(word_list, line_height_threshold):
    """
    Estimate font size based on bounding polygons from Google Vision API.

    Parameters:
    bounding_polys (list): A list of bounding polygons where each polygon is represented
                           as a list of dictionaries with 'x' and 'y' keys for vertices.

    Returns:
    tuple: (min_height, max_height, average_height) based on the heights of bounding boxes.
    """

    heights = []
    current_y = word_list[0][2] if word_list else 0

    total_h = 0
    count = 0
    for word, x, y, h in word_list:
        if is_valid_arabic_word(word):
            total_h += h
            count += 1
        # If the word is on a new line (based on the dynamically calculated threshold), start a new line
        if abs(y - current_y) > line_height_threshold:
            heights.append(total_h/count)
            total_h = 0
            count = 0
            current_y = y

    heights.append(total_h/count)
    print("heights: ", heights)
    avg_height = sum(heights) / len(heights)
    height_std = np.std(heights)

    return avg_height, height_std
