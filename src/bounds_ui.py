import cv2
import matplotlib.pyplot as plt
from numpy import interp

# Global variables to store selected lines
selected_lines = []

def select_lines_callback(event, x, y, flags, param):
    # Mouse callback function to select horizontal lines
    global selected_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_lines) < 3:
            selected_lines.append(y)
            print(selected_lines)

def select_lines_ui(img):
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

    selected_lines[0] = round(interp(selected_lines[0],[0, 500],[0,img.shape[0]]))
    selected_lines[1] = round(interp(selected_lines[1],[0, 500],[0,img.shape[0]]))
    selected_lines[2] = round(interp(selected_lines[2],[0, 500],[0,img.shape[0]]))

    return selected_lines

def display_selections(img, selected_lines):
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
