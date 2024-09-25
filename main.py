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

credentials, project = google.auth.default()
client = vision.ImageAnnotatorClient(credentials=credentials)

def is_valid_arabic_word(word):
    # Define the regex pattern, allowing whitespace characters
    pattern = r'^[\u0600-\u06FF\sA-Za-z$€£]*$'

    # Use re.fullmatch to check if the entire word matches the pattern
    return True if re.fullmatch(pattern, word) else False

def google_ocr(pages):
    for page in pages:
     # Convert the image to bytes
        success, encoded_image = cv2.imencode('.jpg', page)
        if not success:
            print("Error: Could not encode the image.")
            return

        # Prepare image for Google Vision API
        content = encoded_image.tobytes()
        image_for_vision = types.Image(content=content)

         # Add a language hint for Arabic ("ar")
        image_context = vision.ImageContext(language_hints=['ar'])

        # Perform OCR with language hint
        response = client.document_text_detection(image=image_for_vision, image_context=image_context)
        texts = response.text_annotations
        with open('response.json', 'w', encoding='utf-8') as f:
            f.write(str(texts))
        # Display the detected texts

        output_file = "output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            word_list = []
            bounding_polys = []
            if texts:
                print("Detected text. Reordering based on bounding polygons...")

                # Iterate through each word (skip the first annotation since it includes the whole text)
                for word in texts[1:]:
                    # Extract the vertices of the bounding polygon
                    vertices = word.bounding_poly.vertices
                    bounding_polys.append(word.bounding_poly)
                    # Get the x-coordinate of the top-left vertex (for ordering horizontally)
                    top_left_x = vertices[0].x
                    top_left_y = vertices[0].y
                    top_y = min([vertex.y for vertex in vertices])
                    bottom_y = max([vertex.y for vertex in vertices])

                    # Calculate the height of the bounding box
                    height = abs(bottom_y - top_y)
                    # Add the word and its position to the list
                    word_list.append((word.description, top_left_x, top_left_y, height))

                # Calculate the line height threshold dynamically
                line_height_threshold = calculate_line_height_threshold(bounding_polys)
                # print("estimate_font_size_with_min_max", estimate_font_size_with_min_max(bounding_polys))

                # Sort the words: first by y-coordinate (vertical), then by x-coordinate (horizontal)
                word_list.sort(key=lambda w: (w[2], -w[1]))

                # Create an empty list to store the reordered lines
                lines = []
                current_line = []
                heights = []
                current_y = word_list[0][2] if word_list else 0

                total_h = 0
                count = 0
                for word, x, y, h in word_list:
                    # If the word is on a new line (based on the dynamically calculated threshold), start a new line
                    if is_valid_arabic_word(word):
                        total_h += h
                        count += 1
                    if abs(y - current_y) > line_height_threshold:

                        heights.append(total_h/count)
                        total_h = 0
                        count = 0
                        lines.append(current_line)
                        current_line = []
                        current_y = y
                    current_line.append(word)

                # Append the last line
                if current_line:
                    heights.append(total_h/count)
                    lines.append(current_line)
                print(heights)
                avg_height = sum(heights) / len(heights)
                height_std = np.std(heights)
                print("avg height: ", avg_height)
                print("height std: ", height_std)
                # Join words in each line and reverse the lines (because Arabic is right-to-left)
                reordered_text = '\n'.join([' '.join(line) for line in lines])

                # Save the reordered text to the output file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(reordered_text)

                print(f"Reordered text saved to {output_file}.")

        # Handle possible errors
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

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

def estimate_font_size_with_min_max(bounding_polys):
    """
    Estimate font size based on bounding polygons from Google Vision API.

    Parameters:
    bounding_polys (list): A list of bounding polygons where each polygon is represented
                           as a list of dictionaries with 'x' and 'y' keys for vertices.

    Returns:
    tuple: (min_height, max_height, average_height) based on the heights of bounding boxes.
    """
    heights = []

    for bounding_poly in bounding_polys:

        # Extract the y-coordinates of the top and bottom vertices
        top_y = min([vertex.y for vertex in bounding_poly.vertices])
        bottom_y = max([vertex.y for vertex in bounding_poly.vertices])

        # Calculate the height of the bounding box
        height = abs(bottom_y - top_y)
        heights.append(height)

    if not heights:
        return (0, 0, 0)  # Return 0s if no valid bounding boxes

    min_height = min(heights)
    max_height = max(heights)
    avg_height = sum(heights) / len(heights)

    return (min_height, max_height, avg_height)

# Global variables to store selected lines
selected_lines = []

def select_lines_callback(event, x, y, flags, param):
    # Mouse callback function to select horizontal lines
    global selected_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_lines) < 3:
            selected_lines.append(y)
            print(selected_lines)

def select_lines_ui():
    # Load the document image
    img = cv2.imread('output_mksa.png')

    if img is None:
        print("Error: Image not found!")
        return

    # Resize image for better viewing (optional)
    img_resized = cv2.resize(img, (400, 500))

    # Show instructions
    print("Select 3 horizontal lines: for Header, Main Body, and Footer")

    # Create a window to display the image
    cv2.namedWindow('Document')
    cv2.setMouseCallback('Document', select_lines_callback)

    # Show the image
    while True:
        cv2.imshow('Document', img_resized)
        if cv2.waitKey(1) & 0xFF == 27:  # Escape key to close
            break

    # Sort the selected lines
    selected_lines.sort()

    if len(selected_lines) != 3:
        print("Error: 3 lines not selected.")
        return

    # Draw the selected lines on the image
    for line_y in selected_lines:
        cv2.line(img_resized, (0, line_y), (img_resized.shape[1], line_y), (255, 0, 0), 2)

    print("img.shape[:2]", img.shape[0])
    print("interp", round(interp(selected_lines[0],[0, 500],[0,img.shape[0]])))
    selected_lines[0] = round(interp(selected_lines[0],[0, 500],[0,img.shape[0]]))
    selected_lines[1] = round(interp(selected_lines[1],[0, 500],[0,img.shape[0]]))
    selected_lines[2] = round(interp(selected_lines[2],[0, 500],[0,img.shape[0]]))
    print("selected_lines", selected_lines)
    # Create separate sections for Header, Main Body, and Footer
    header = img[0:selected_lines[0], :]
    body = img[selected_lines[0]:selected_lines[1], :]
    footnotes = img[selected_lines[1]:selected_lines[2], :]
    footer = img[selected_lines[2]:, :]

    # Display the sections
    plt.figure(figsize=(10, 10))

    plt.subplot(4, 1, 1)
    plt.imshow(cv2.cvtColor(header, cv2.COLOR_BGR2RGB))
    plt.title('Header')

    plt.subplot(4, 1, 2)
    plt.imshow(cv2.cvtColor(body, cv2.COLOR_BGR2RGB))
    plt.title('Main Body')

    plt.subplot(4, 1, 3)
    plt.imshow(cv2.cvtColor(footnotes, cv2.COLOR_BGR2RGB))
    plt.title('Footnotes')

    plt.subplot(4, 1, 4)
    plt.imshow(cv2.cvtColor(footer, cv2.COLOR_BGR2RGB))
    plt.title('Footer')

    plt.tight_layout()
    plt.show()

    return body, footnotes


if __name__ == "__main__":
    print("main")
    # pages = convert_from_path('mksa.pdf', dpi=400)

    # first_page = preprocess_image(pages[12])

    # # Save the image as a PNG file
    # first_page.save("output_mksa.png", "PNG")

    body, footnotes = select_lines_ui()
    pages = [body, footnotes]
    google_ocr(pages)

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
