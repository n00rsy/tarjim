from pdf2image import convert_from_path
import cv2
import numpy as np
from bounds_ui import select_lines_ui, display_selections
from ocr import google_ocr, find_footnote_line

if __name__ == "__main__":
    pages = convert_from_path('../mksa.pdf', dpi=400)
    start = 9
    end = 46
    # Save the image as a PNG file
    # first_page.save("output_mksa.png", "PNG")
    # img = cv2.imread('../output_mksa.png')

    #selected_lines = select_lines_ui(img)

    # display_selections(img, selected_lines)
    selected_lines = [491, 2608, 3467]
    all_text = ""
    for i, page_pil in enumerate(pages[start:end]):
        print(f"processing page {i}")
        # Convert RGB to BGR (OpenCV format)
        page_cv2 = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
        page_h, page_w, *_ = page_cv2.shape
        # do OCR to find footer line
        footnote_line = find_footnote_line(page_cv2, selected_lines[0], i)
        print("footnote_line: ", footnote_line)
        body = page_cv2[selected_lines[0]:footnote_line, :]
        footnotes = page_cv2[footnote_line:selected_lines[2], :]

        cv2.imwrite(f"temp/output_{i}_ocr_body.png", body)
        if footnote_line < page_h:
            cv2.imwrite(f"temp/output_{i}_ocr_footnote.png", footnotes)

        all_text = all_text + '\n' + google_ocr(body)

        print("\n")

    f = open("temp/mksa_output.txt", "w")
    f.write(all_text)
    f.close()
    # chatgpt
