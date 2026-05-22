import cv2
import pandas as pd
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open webcam
cap = cv2.VideoCapture(0)

attendance_marked = False

print("Starting Face Detection Attendance System...")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Mark attendance once
        if not attendance_marked:
            now = datetime.now()
            time_string = now.strftime("%H:%M:%S")

            attendance = pd.DataFrame({
                "Name": ["Detected_User"],
                "Status": [f"Present at {time_string}"]
            })

            attendance.to_csv(
                "attendance.csv",
                mode='a',
                header=False,
                index=False
            )

            attendance_marked = True

    cv2.imshow("Face Detection Attendance System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Attendance saved successfully.")