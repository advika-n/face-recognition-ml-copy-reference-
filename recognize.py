import os
import cv2
import pandas as pd
import requests
import threading

# -------------------------------------------------------
# recognize.py
# Opens camera, recognizes faces, notifies server.py
# which then marks attendance on Railway backend.
# -------------------------------------------------------

DISPLAY_SERVER = "http://localhost:5050"

def notify_display(name, reg_no, department, confidence_pct, classroom):
    """Tell server.py a face was detected — runs in background thread"""
    try:
        requests.post(f"{DISPLAY_SERVER}/detected", json={
            "name": name,
            "reg_no": reg_no,
            "department": department,
            "confidence": confidence_pct,
            "classroom": classroom
        }, timeout=2)
    except Exception as e:
        print(f"Display server not reachable: {e}")

def clear_display_after_delay(seconds=3):
    """Wait then reset display — runs in background so camera doesn't freeze"""
    import time
    time.sleep(seconds)
    try:
        requests.post(f"{DISPLAY_SERVER}/clear", timeout=2)
    except:
        pass

def recognize_attendance():
    # Load trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./TrainingImageLabel/Trainner.yml")

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    # Load student details CSV — ensure Id column is integer for matching
    df = pd.read_csv("StudentDetails" + os.sep + "StudentDetails.csv")
    df['Id'] = df['Id'].astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX

    classroom = input("Enter Classroom (e.g. 301): ").strip()

    # Track who has already been marked this session
    marked = set()

    # Start camera — try index 0 then 1
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("ERROR: Could not open camera.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("\nRecognizing faces... Press 'q' to stop.\n")

    while True:
        _, im = cam.read()
        if im is None:
            continue

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, 1.2, 5,
            minSize=(int(minW), int(minH)),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            confidence_pct = round(100 - conf)

            if conf < 100:
                # Id from recognizer is int, CSV Id column is also int now
                student_row = df.loc[df['Id'] == Id]
                if not student_row.empty:
                    name = student_row['Name'].values[0]
                    reg_no = str(student_row['RegistrationNumber'].values[0])
                    department = str(student_row['Department'].values[0]) if 'Department' in df.columns else ''
                else:
                    name = "Unknown"
                    reg_no = None
                    department = ''
            else:
                name = "Unknown"
                reg_no = None
                department = ''

            # Mark if confidence > 67% and not already marked
            if confidence_pct > 67 and reg_no and reg_no not in marked:
                marked.add(reg_no)
                print(f"Detected: {name} ({reg_no}) — {confidence_pct}% confidence")

                # Notify display in background (doesn't freeze camera)
                threading.Thread(
                    target=notify_display,
                    args=(name, reg_no, department, confidence_pct, classroom),
                    daemon=True
                ).start()

                # Clear display after 3 seconds in background
                threading.Thread(
                    target=clear_display_after_delay,
                    args=(3,),
                    daemon=True
                ).start()

                label = f"{name} [Marked]"

            elif confidence_pct > 67 and reg_no:
                label = f"{name} [Already Marked]"
            else:
                label = "Unknown"

            cv2.putText(im, label, (x+5, y-5), font, 1, (255, 255, 255), 2)

            conf_text = f"{confidence_pct}%"
            if confidence_pct > 67:
                colour = (0, 255, 0)
            elif confidence_pct > 50:
                colour = (0, 255, 255)
            else:
                colour = (0, 0, 255)
            cv2.putText(im, conf_text, (x+5, y+h-5), font, 1, colour, 1)

        cv2.imshow('Attendance', im)

        if cv2.waitKey(1) == ord('q'):
            break

    print(f"\nSession ended. {len(marked)} student(s) marked present.")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_attendance()
