import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # For timezone-aware datetime (Python 3.9+)

# Path to your service account key JSON file
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

def add_dummy_session(user_id):
    attendance_ref = db.collection('attendance').document(user_id).collection('sessions')
    # Set timezone to Asia/Kolkata for correct IST time
    check_in = datetime(2025, 6, 7, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))  # 9:00 AM IST
    check_out = check_in + timedelta(hours=1)  # 6 hours later

    duration = int((check_out - check_in).total_seconds() // 60)

    session_data = {
        'checkInLocation': 'KLE Tech Campus',
        'checkInTimestamp': check_in,
        'checkOutLocation': 'KLE Tech Campus',
        'checkOutTimestamp': check_out,
        'duration': duration,
    }

    attendance_ref.add(session_data)
    print('Dummy session added!')

# Replace with your actual user ID
add_dummy_session('hlhRy7IaRLbH1fJ1fzZyiRLsbpm1')