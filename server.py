from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

BACKEND_URL = "https://facial-recognition-attendance-backend-production.up.railway.app/api/mark-attendance/"

# Holds the latest detection state
latest = {
    "status": "waiting",
    "name": "",
    "reg_no": "",
    "department": "",
    "confidence": 0
}

@app.route('/status')
def status():
    """HTML file polls this every second to get latest detection"""
    return jsonify(latest)

@app.route('/detected', methods=['POST'])
def detected():
    """recognize.py calls this when a face is identified"""
    global latest
    data = request.json

    latest = {
        "status": "detected",
        "name": data['name'],
        "reg_no": data['reg_no'],
        "department": data.get('department', ''),
        "confidence": data.get('confidence', 0)
    }

    # Forward attendance to Railway backend
    try:
        response = requests.post(BACKEND_URL, json={
            "registration_number": data['reg_no'],
            "classroom": data['classroom']
        })
        if response.status_code in [200, 201]:
            latest["backend_status"] = "success"
            print(f"✓ Attendance marked for {data['reg_no']}")
        else:
            latest["backend_status"] = "failed"
            print(f"✗ Backend error for {data['reg_no']}: {response.text}")
    except Exception as e:
        latest["backend_status"] = "failed"
        print(f"✗ Could not reach backend: {e}")

    return jsonify({"ok": True})

@app.route('/clear', methods=['POST'])
def clear():
    """recognize.py calls this after a few seconds to reset the display"""
    global latest
    latest = {
        "status": "waiting",
        "name": "",
        "reg_no": "",
        "department": "",
        "confidence": 0
    }
    return jsonify({"ok": True})

if __name__ == '__main__':
    print("Display server running on http://localhost:5050")
    app.run(port=5050)
