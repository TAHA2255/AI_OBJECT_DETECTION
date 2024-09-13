import os
from ultralytics import YOLO
import cv2

# Define image path
IMAGES_DIR = os.path.join('.', 'images')
image_path = os.path.join(IMAGES_DIR, 'img1.jpg')
image_path_out = '{}_out.jpg'.format(image_path)

# Load the image
image = cv2.imread(image_path)
H, W, _ = image.shape

# Load the model
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Perform the prediction
results = model(image)[0]

# Process results
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Draw bounding box and label
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the output image
cv2.imwrite(image_path_out, image)

# Display the output image (optional)
cv2.imshow('Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
