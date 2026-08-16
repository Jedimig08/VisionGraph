import discovery
import cv2

url = f"{discovery.app_url()}/camera/0"

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