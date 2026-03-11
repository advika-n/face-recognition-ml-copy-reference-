import cv2
import os
import csv

# -------------------------------------------------------
# capture.py
# Run once per student to capture their face images.
# Images saved to TrainingImage/
# Student details saved to StudentDetails/StudentDetails.csv
# -------------------------------------------------------

def capture_faces():
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    student_id = input("Enter Student Numeric ID (e.g. 1, 2, 3...): ").strip()
    student_name = input("Enter Student Name: ").strip()
    registration_number = input("Enter Registration Number (e.g. 21BCE1234): ").strip()

    if not os.path.exists("TrainingImage"):
        os.makedirs("TrainingImage")
    if not os.path.exists("StudentDetails"):
        os.makedirs("StudentDetails")

    # Save to CSV
    csv_path = "StudentDetails" + os.sep + "StudentDetails.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Id', 'Name', 'RegistrationNumber'])
        # Save Id as integer string so it matches recognizer.predict() output
        writer.writerow([int(student_id), student_name, registration_number])

    # Start camera — try index 0 then 1
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("ERROR: Could not open camera. Check that your camera is connected.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    sample_count = 0
    total_samples = 100

    print(f"\nCapturing face images for {student_name}. Look at the camera...")
    print("Press 'q' to quit early.\n")

    while True:
        ret, img = cam.read()
        if not ret or img is None:
            print("WARNING: Could not read frame from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_count += 1
            filename = f"TrainingImage{os.sep}User.{student_id}.{sample_count}.jpg"
            cv2.imwrite(filename, gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
            cv2.putText(img, f"Capturing: {sample_count}/{total_samples}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Capturing Faces', img)

        if cv2.waitKey(1) == ord('q') or sample_count >= total_samples:
            break

    print(f"\nDone! {sample_count} images captured for {student_name}.")
    print("Now run train.py to retrain the model.")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_faces()
