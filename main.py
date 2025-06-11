import cv2
import os
from deepface import DeepFace
from numpy import dot
from numpy.linalg import norm
import threading
from datetime import datetime
import pytz

from google.cloud import firestore
from google.oauth2 import service_account

# ------------------ Firestore Setup ------------------
key_path = os.path.join(os.getcwd(), "serviceAccountKey.json")
credentials = service_account.Credentials.from_service_account_file(key_path)
db = firestore.Client(credentials=credentials)

# ------------------ Load All Reference Embeddings ------------------
reference_embeddings = {}

REFERENCE_FOLDER = "reference_images"  # Folder where UID-named images are stored

for filename in os.listdir(REFERENCE_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        uid = os.path.splitext(filename)[0]
        img_path = os.path.join(REFERENCE_FOLDER, filename)
        try:
            embedding = DeepFace.represent(img_path, enforce_detection=False)[0]["embedding"]
            reference_embeddings[uid] = embedding
            print(f"Loaded embedding for UID: {uid}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# ------------------ Helper Functions ------------------
def find_matching_uid(face_img):
    try:
        resized = cv2.resize(face_img, (160, 160))
        candidate_embedding = DeepFace.represent(resized, enforce_detection=False)[0]["embedding"]

        for uid, ref_embedding in reference_embeddings.items():
            similarity = dot(candidate_embedding, ref_embedding) / (norm(candidate_embedding) * norm(ref_embedding))
            if similarity > 0.7:  # similarity threshold
                return uid
    except Exception as e:
        print(f"Error in find_matching_uid: {e}")
    return None

def auto_checkout(user_uid):
    print(f"🔍 Looking up active session for user UID: {user_uid}")

    sessions_ref = db.collection("attendance").document(user_uid).collection("sessions")
    sessions = sessions_ref.stream()

    active_session = None
    for session in sessions:
        data = session.to_dict()
        # Active session: checkOutTimestamp missing or None
        if "checkOutTimestamp" not in data or data["checkOutTimestamp"] is None:
            active_session = session
            break

    if not active_session:
        print(f"⚠️ No active session found for user: {user_uid}")
        return False

    data = active_session.to_dict()

    print(f"🟢 Active Session ID: {active_session.id}")
    print(f"  🔸 CheckInTimestamp: {data.get('checkInTimestamp')}")
    print(f"  🔸 CheckInLocation: {data.get('checkInLocation')}")

    now = datetime.now(pytz.utc)
    check_in_time = data.get("checkInTimestamp")

    if not check_in_time:
        print("❌ Error: checkInTimestamp missing in session data.")
        return False

    duration_minutes = int((now - check_in_time).total_seconds() // 60)
    if duration_minutes < 0:
        duration_minutes = 0

    # Update checkout info
    session_ref = sessions_ref.document(active_session.id)
    session_ref.update({
        "checkOutTimestamp": now,
        "checkOutLocation": "KLE Tech Campus",
        "duration": duration_minutes
    })

    print("✅ Successfully auto checked out.")
    print(f"  🔸 CheckOutLocation: KLE Tech Campus")
    print(f"  🔸 CheckOutTimestamp: {now}")
    print(f"  🔸 Duration (minutes): {duration_minutes}")
    return True

# ------------------ Face Recognition Logic ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam could not be opened! Try changing the index (0, 1, 2) or check permissions.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognition_lock = threading.Lock()
recognized_uid = None
recognition_thread = None

def recognize_and_checkout(face_img):
    global recognized_uid
    uid = find_matching_uid(face_img)
    with recognition_lock:
        recognized_uid = uid
    if uid:
        auto_checkout(uid)

def start_recognition(face_img):
    global recognition_thread
    if recognition_thread is None or not recognition_thread.is_alive():
        recognition_thread = threading.Thread(target=recognize_and_checkout, args=(face_img,))
        recognition_thread.start()

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    label = "No face matched"
    color = (0, 0, 255)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x1, y1, w1, h1 = x*2, y*2, w*2, h*2
        face_img = frame[y1:y1+h1, x1:x1+w1]

        start_recognition(face_img)

        with recognition_lock:
            if recognized_uid:
                label = f"User {recognized_uid} recognized and checked out"
                color = (0, 255, 0)
            else:
                label = "Recognizing..."
                color = (0, 255, 255)
    else:
        with recognition_lock:
            recognized_uid = None
        label = "No face detected"
        color = (0, 0, 255)

    # Draw rectangle and label
    for (x, y, w, h) in faces:
        x1, y1, w1, h1 = x*2, y*2, w*2, h*2
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), color, 2)

    cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Auto Face Checkout", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
