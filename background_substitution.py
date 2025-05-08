import cv2
import numpy as np

# --- Configuration ---
VIRTUAL_BG_PATH = "moon.jfif"
CAMERA_ID = 0
BG_TRAINING_FRAMES = 60
MOG2_HISTORY = 100
LEARNING_RATE = 0.01

# --- Load virtual background image ---
virtual_bg = cv2.imread(VIRTUAL_BG_PATH)
if virtual_bg is None:
    raise IOError(f"Cannot load background image at '{VIRTUAL_BG_PATH}'")

# --- Initialize webcam ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# --- Create background subtractor ---
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=50, detectShadows=False)

# --- Morphological kernel ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- Background training ---
print("Training background model. Please stay out of frame...")
frame_count = 0
while frame_count < BG_TRAINING_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    bg_subtractor.apply(frame, learningRate=LEARNING_RATE)
    frame_count += 1
    cv2.putText(frame, "Training Background... Stay out of frame", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Training")
print("Background model trained. Starting background substitution...")

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    bg_resized = cv2.resize(virtual_bg, (w, h))

    # Apply background subtractor
    fg_mask = bg_subtractor.apply(frame, learningRate=0)

    # Clean up the mask
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Convert to 3 channels
    fg_mask_3ch = cv2.merge([fg_mask] * 3)
    bg_mask_3ch = cv2.bitwise_not(fg_mask_3ch)

    # Optional: feathered mask for blending
    fg_blend_mask = cv2.GaussianBlur(fg_mask_3ch.astype(float) / 255.0, (15, 15), 0)

    # Apply masks with blending
    frame_float = frame.astype(float)
    bg_float = bg_resized.astype(float)

    result = (frame_float * fg_blend_mask + bg_float * (1 - fg_blend_mask)).astype(np.uint8)

    # Combine views
    comparison = np.hstack((frame, result))
    cv2.imshow("Webcam | Background Substitution", comparison)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
