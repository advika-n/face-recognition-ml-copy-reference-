import cv2
import numpy as np
import requests
import base64
import threading
import time

# -------------------------------------------------------
# recognize.py
# Fetches face encodings from Railway at startup.
# Opens camera, recognizes faces, requires a blink for
# liveness detection, then POSTs attendance to backend.
# -------------------------------------------------------

BACKEND_URL = "https://facial-recognition-attendance-backend-production.up.railway.app"
DISPLAY_SERVER = "http://localhost:5050"

# ── Blink detection tuning ─────────────────────────────
# EAR = Eye Aspect Ratio. When your eye is open, EAR ≈ 0.3.
# When you blink, EAR drops below 0.25 for a few frames.
EAR_THRESHOLD = 0.15       # below this = eye is closed (bbox ratio scale)
EAR_CONSEC_FRAMES = 2      # must be closed for this many frames to count as a blink
BLINK_REQUIRED = 1         # number of blinks needed before marking attendance
# ───────────────────────────────────────────────────────


def eye_aspect_ratio(eye_points):
    """
    Calculate EAR (Eye Aspect Ratio) from 6 eye landmark points.

    The eye landmark points form a shape like this:
         1  2
        0    3
         5  4
    EAR = (|1-5| + |2-4|) / (2 * |0-3|)
    When eye is open, the vertical distances (1-5, 2-4) are large → EAR ≈ 0.3
    When eye is closed, vertical distances collapse → EAR ≈ 0.0
    """
    # Vertical distances
    v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    # Horizontal distance
    h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def get_ear_from_landmarks(landmarks):
    """
    Calculate Eye Aspect Ratio from face_recognition landmarks.
    Instead of relying on point ordering (which varies), we use the
    bounding box of all eye points:
      EAR = eye height / eye width
    When open: ~0.3  |  When closed: ~0.05
    """
    left_eye  = landmarks.get("left_eye",  [])
    right_eye = landmarks.get("right_eye", [])

    if len(left_eye) < 4 or len(right_eye) < 4:
        return None

    def bbox_ear(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        width  = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width == 0:
            return 0.0
        return height / width

    return (bbox_ear(left_eye) + bbox_ear(right_eye)) / 2.0


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
            encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
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
        import face_recognition
    except ImportError:
        print("✗ face_recognition not installed. Run: pip install face-recognition")
        return

    known_encodings, known_students = load_encodings_from_backend()

    if not known_encodings:
        print("✗ No face encodings found. Register student faces first via the admin panel.")
        return

    classroom = input("Enter Classroom (e.g. 301): ").strip()
    marked = set()         # reg numbers fully marked (blink confirmed)
    pending = {}           # reg numbers recognized but awaiting blink
                           # { reg_no: { "blink_count": int, "ear_consec": int, "eye_was_open": bool } }

    # Start camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("✗ Could not open camera.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    print(f"\nRecognizing faces for classroom {classroom}...")
    print("Students must BLINK to confirm liveness before attendance is marked.")
    print("Press 'q' to stop.\n")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # Resize to 50% for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings_list = face_recognition.face_encodings(rgb_small, face_locations)

        # Get facial landmarks for EAR calculation (same scale as small_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small, face_locations)

        for face_encoding, face_location, landmarks in zip(face_encodings_list, face_locations, face_landmarks_list):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(np.argmin(distances))
            best_distance = distances[best_idx]
            confidence_pct = round((1 - best_distance) * 100, 1)

            top, right, bottom, left = [v * 2 for v in face_location]

            if best_distance >= 0.6:
                # Unknown face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left + 5, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            student = known_students[best_idx]
            name = student["name"]
            reg_no = student["registration_number"]
            department = student["department"]

            if reg_no in marked:
                # Already fully confirmed — just show label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                cv2.putText(frame, f"{name} [Already Marked]", (left + 5, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                continue

            # ── Blink detection ───────────────────────────────────
            ear = get_ear_from_landmarks(landmarks)

            if reg_no not in pending:
                pending[reg_no] = {"blink_count": 0, "ear_consec": 0, "eye_was_open": True}
                print(f"  ◉ Recognized: {name} ({reg_no}) — {confidence_pct}% confidence. Please BLINK to confirm.")

            state = pending[reg_no]

            if ear is not None:
                if ear < EAR_THRESHOLD:
                    # Eye appears closed this frame
                    state["ear_consec"] += 1
                    state["eye_was_open"] = False
                else:
                    # Eye is open
                    if not state["eye_was_open"] and state["ear_consec"] >= EAR_CONSEC_FRAMES:
                        # Eye just re-opened after being closed long enough → blink detected
                        state["blink_count"] += 1
                        print(f"  👁 Blink {state['blink_count']}/{BLINK_REQUIRED} detected for {name}")
                    state["ear_consec"] = 0
                    state["eye_was_open"] = True

            blinks_remaining = BLINK_REQUIRED - state["blink_count"]

            if state["blink_count"] >= BLINK_REQUIRED:
                # Liveness confirmed — mark attendance
                marked.add(reg_no)
                del pending[reg_no]
                print(f"✓ Liveness confirmed! Attendance marked for {name} ({reg_no})")
                threading.Thread(target=mark_attendance_api, args=(reg_no, classroom), daemon=True).start()
                threading.Thread(target=notify_display, args=(name, reg_no, department, confidence_pct, classroom), daemon=True).start()
                threading.Thread(target=clear_display_after_delay, args=(3,), daemon=True).start()
                colour = (0, 255, 0)
                label = f"{name} [Marked] {confidence_pct}%"
            else:
                colour = (0, 165, 255)  # orange = waiting for blink
                label = f"{name} — Blink {blinks_remaining}x to confirm"

            cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
            cv2.putText(frame, label, (left + 5, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            # Show EAR value in corner for debugging (remove for production)
            if ear is not None:
                cv2.putText(frame, f"EAR: {ear:.2f}", (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Attendance Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\nSession ended. {len(marked)} student(s) marked present.")
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_attendance()
