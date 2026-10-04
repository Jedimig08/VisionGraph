import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import discovery
import cv2

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

url = f"{discovery.app_url()}/stream/1"

print(f"Connecting to {url}...")

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't read stream")
        break
    cv2.imshow("Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()