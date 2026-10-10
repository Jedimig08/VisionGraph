import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import discovery
import camera_select

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

camera = camera_select.select_camera()
fps = camera_select.MODERATE_FPS
width, height = camera_select.choose_resolution(camera)

url = f"{discovery.app_url()}/stream/{camera['id']}?res={width}x{height}&fps={fps}"

print(
    f"Selected camera {camera['id']} "
    f"({camera.get('facing')} {camera.get('type')}) "
    f"at {width}x{height} @ {fps} FPS"
)
print(f"Connecting to {url}...")

cap = cv2.VideoCapture(url)
cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
fullscreen = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't read stream")
        break

    cv2.imshow("Stream", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            "Stream",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )

cap.release()
cv2.destroyAllWindows()