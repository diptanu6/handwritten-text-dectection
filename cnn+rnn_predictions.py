import numpy as np
import cv2
from tensorflow.keras.models import load_model  # type: ignore
from imutils.contours import sort_contours
import imutils

def predict_words():
    image_path ="dip.jpg"
    model_path = "output_model.h5"

    print("[INFO] Loading model...")
    model = load_model(model_path)

    print("[INFO] Preprocessing image...")
    image = cv2.imread(image_path)

    if image is None:
        print("[ERROR] Unable to read the image file. Please check the path.")
        return

    # Save original image
    cv2.imwrite("step_0_original.png", image)
    print("[INFO] Saved original image as 'step_0_original.png'.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("step_1_grayscale.png", gray)
    print("[INFO] Saved grayscale image as 'step_1_grayscale.png'.")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("step_2_blurred.png", blurred)
    print("[INFO] Saved blurred image as 'step_2_blurred.png'.")

    edged = cv2.Canny(blurred, 30, 150)
    cv2.imwrite("step_3_edged.png", edged)
    print("[INFO] Saved edged image as 'step_3_edged.png'.")

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    chars = []
    char_images = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            char_images.append(thresh)  # Append the thresholded image for collage

            tH, tW = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

            tH, tW = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            padded = cv2.copyMakeBorder(
                thresh, top=dY, bottom=dY, left=dX, right=dX,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

            padded = cv2.resize(padded, (32, 32))
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            chars.append((padded, (x, y, w, h)))

    if not chars:
        print("[INFO] No valid characters found.")
        return

    # Combine all thresholded character images into a single horizontal image
    if char_images:
        max_height = max(img.shape[0] for img in char_images)
        resized_char_images = [
            cv2.copyMakeBorder(img, 0, max_height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
            for img in char_images
        ]
        combined_image = np.hstack(resized_char_images)

        cv2.imwrite("step_4_combined_thresholded_characters.png", combined_image)
        print("[INFO] Saved combined thresholded characters as 'step_4_combined_thresholded_characters.png'.")

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    print("[INFO] Predicting characters...")
    preds = model.predict(chars)
    label_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_names = [l for l in label_names]

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        prob = max(pred)
        label = label_names[np.argmax(pred)]

        print(f"[INFO] {label} - {prob * 100:.2f}%")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Save final image with annotations
    cv2.imwrite("step_5_cnn+rnn.png", image)
    print("[INFO] Saved final annotated image as 'step_5_final_output.png'.")

if __name__ == "__main__":
    predict_words()
