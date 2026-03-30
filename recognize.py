import cv2
import numpy as np
import requests
import base64
import threading
import time

# -------------------------------------------------------
# recognize.py
# Fetches face encodings from Railway at startup.
# Opens camera, recognizes faces, POSTs attendance to backend.
# Optionally notifies server.py for display monitor.
# -------------------------------------------------------

BACKEND_URL = "https://facial-recognition-attendance-backend-production.up.railway.app"
DISPLAY_SERVER = "http://localhost:5050"


def load_encodings_from_backend():
    """Fetch all face encodings from Railway at startup."""
    print("Loading face encodings from backend...")
    try:
        res = requests.get(f"{BACKEND_URL}/api/get-encodings/", timeout=10)
        data = res.json()
        known_encodings = []
        known_students = []
        for entry in data.get("encodings", []):
            encoding_bytes = base64.b64decode(entry["encoding"])
            # InsightFace uses 512-d float32 (was float64 for dlib)
            encoding = np.frombuffer(encoding_bytes, dtype=np.float32)
            known_encodings.append(encoding)
            known_students.append({
                "name": entry["name"],
                "registration_number": entry["registration_number"],
                "department": entry.get("department", "")
            })
        print(f"✓ Loaded {len(known_encodings)} face encoding(s).")
        return known_encodings, known_students
    except Exception as e:
        print(f"✗ Failed to load encodings: {e}")
        return [], []


def mark_attendance_api(reg_no, classroom):
    """POST to Railway to record attendance."""
    try:
        res = requests.post(f"{BACKEND_URL}/api/mark-attendance/", json={
            "registration_number": reg_no,
            "classroom": classroom
        }, timeout=5)
        if res.status_code in [200, 201]:
            print(f"  → Attendance marked for {reg_no}")
        else:
            print(f"  → Backend: {res.json().get('error', 'unknown error')}")
    except Exception as e:
        print(f"  → Could not reach backend: {e}")


def notify_display(name, reg_no, department, confidence_pct, classroom):
    """Notify local display server (optional)."""
    try:
        requests.post(f"{DISPLAY_SERVER}/detected", json={
            "name": name, "reg_no": reg_no,
            "department": department, "confidence": confidence_pct,
            "classroom": classroom
        }, timeout=2)
    except:
        pass


def clear_display_after_delay(seconds=3):
    time.sleep(seconds)
    try:
        requests.post(f"{DISPLAY_SERVER}/clear", timeout=2)
    except:
        pass


def recognize_attendance():
    try:
        import insightface
    except ImportError:
        print("✗ insightface not installed. Run: pip install insightface onnxruntime")
        return

    known_encodings, known_students = load_encodings_from_backend()

    if not known_encodings:
        print("✗ No face encodings found. Register student faces first via the admin panel.")
        return

    classroom = input("Enter Classroom (e.g. 301): ").strip()
    marked = set()

    # Load InsightFace ArcFace model once at startup
    print("Loading InsightFace model...")
    app = insightface.app.FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    print("✓ Model ready.")

    # Start camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("✗ Could not open camera.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    print(f"\nRecognizing faces for classroom {classroom}... Press 'q' to stop.\n")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # InsightFace works directly on BGR — no need to convert to RGB
        faces = app.get(frame)

        for face in faces:
            query_enc = face.normed_embedding.astype(np.float32)

            # Cosine similarity — higher means more similar (opposite of dlib)
            similarities = [float(np.dot(query_enc, k)) for k in known_encodings]
            best_idx = int(np.argmax(similarities))
            best_sim = similarities[best_idx]
            confidence_pct = round(best_sim * 100, 1)

            # Get bounding box from InsightFace
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

            # ArcFace threshold: >0.35 strong match, <0.20 clearly unknown
            THRESHOLD = 0.35
            if best_sim > THRESHOLD:
                student = known_students[best_idx]
                name = student["name"]
                reg_no = student["registration_number"]
                department = student["department"]
                colour = (0, 255, 0)

                if reg_no not in marked:
                    marked.add(reg_no)
                    print(f"✓ Detected: {name} ({reg_no}) — {confidence_pct}% confidence")

                    threading.Thread(target=mark_attendance_api, args=(reg_no, classroom), daemon=True).start()
                    threading.Thread(target=notify_display, args=(name, reg_no, department, confidence_pct, classroom), daemon=True).start()
                    threading.Thread(target=clear_display_after_delay, args=(3,), daemon=True).start()

                    label = f"{name} [Marked] {confidence_pct}%"
                else:
                    label = f"{name} [Already Marked]"
                    colour = (0, 255, 255)
            else:
                label = "Unknown"
                colour = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
            cv2.putText(frame, label, (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imshow("Attendance Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nSession ended. {len(marked)} student(s) marked present.")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_attendance()
