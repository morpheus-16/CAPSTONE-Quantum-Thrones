import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="0muhNzus6VvhbWfFeNp6"
)

# Capture one frame from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite("test_frame.jpg", frame)
    result = CLIENT.infer("test_frame.jpg", model_id="american-sign-language-letters/6")
    print(result)
else:
    print("Camera not working")