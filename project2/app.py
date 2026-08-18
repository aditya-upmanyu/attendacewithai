"""
Smart Attendance System - GLA University
Google Solution Challenge 2026
Team: Consciousness 

AI-Powered Facial Recognition Attendance System
Features: Face Recognition, Real-time Attendance, Analytics, Gemini AI Insights
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime, timezone, timedelta
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import logging
import threading
import time
from functools import wraps
import json

# Load environment variables
load_dotenv()

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('attendance_system.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ==================== FLASK APPLICATION SETUP ====================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "gla_smart_attendance_system_2026_secure_key")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=12)
app.config['SESSION_COOKIE_SECURE'] = False  # Set True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==================== FIREBASE INITIALIZATION ====================
try:
    if not os.path.exists("serviceAccountKey.json"):
        raise FileNotFoundError("❌ serviceAccountKey.json not found! Please add Firebase credentials.")
    
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")
    })
    
    bucket = storage.bucket()
    logger.info("✅ Firebase Realtime Database initialized successfully")
    logger.info("✅ Firebase Storage initialized successfully")
    
except Exception as e:
    logger.critical(f"❌ Firebase initialization FAILED: {e}")
    raise SystemExit("Cannot start without Firebase. Please check configuration.")

# ==================== OPENCV FACE DETECTION ====================
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise ValueError("Failed to load Haar Cascade classifier")
    
    logger.info("✅ OpenCV face detection loaded successfully")
    
except Exception as e:
    logger.critical(f"❌ OpenCV initialization FAILED: {e}")
    raise SystemExit("Cannot start without OpenCV. Please install opencv-python.")

# ==================== GOOGLE GEMINI AI INTEGRATION ====================
gemini_model = None
try:
    import google.generativeai as genai
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip():
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("✅ Google Gemini AI initialized successfully")
    else:
        logger.warning("⚠️ GEMINI_API_KEY not found in .env - AI insights will be disabled")
        
except ImportError:
    logger.warning("⚠️ google-generativeai not installed - AI insights disabled")
except Exception as e:
    logger.warning(f"⚠️ Gemini AI initialization failed: {e} - AI insights disabled")

# ==================== AUTHENTICATION DECORATOR ====================
def login_required(f):
    """Decorator to protect routes requiring authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session.get('logged_in'):
            flash('Please log in to access this page', 'error')
            logger.warning(f"Unauthorized access attempt to {request.path}")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== ATTENDANCE CACHE SYSTEM ====================
class AttendanceCache:
    """
    Thread-safe cache for real-time face recognition during attendance sessions.
    Stores student face encodings and tracks marked attendance.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.encodings = {}      # {student_id: numpy_array}
        self.info = {}           # {student_id: {name, roll_no}}
        self.marked = set()      # Set of marked student IDs
        self.last_seen = {}      # {student_id: timestamp}
        self.cooldown = 3        # Cooldown period in seconds
    
    def load_class(self, class_id):
        """Load all students from a class into memory for fast matching"""
        with self.lock:
            # Clear previous data
            self.encodings.clear()
            self.info.clear()
            self.marked.clear()
            self.last_seen.clear()
            
            try:
                class_data = db.reference(f'classes/{class_id}').get()
                if not class_data:
                    logger.warning(f"⚠️ Class {class_id} not found in database")
                    return 0
                
                students = class_data.get('students', {})
                
                for student_id, student_data in students.items():
                    encoding_list = student_data.get('encoding', [])
                    
                    if encoding_list and len(encoding_list) > 0:
                        encoding_array = np.array(encoding_list)
                        self.encodings[student_id] = encoding_array
                        self.info[student_id] = {
                            'name': student_data.get('name', 'Unknown'),
                            'roll_no': student_data.get('roll_no', 'N/A')
                        }
                
                logger.info(f"📚 Successfully loaded {len(self.encodings)} students into cache for class {class_id}")
                return len(self.encodings)
                
            except Exception as e:
                logger.error(f"❌ Error loading class {class_id} into cache: {e}")
                return 0
    
    def can_mark(self, student_id):
        """Check if student can be marked (respects cooldown period)"""
        with self.lock:
            # Already marked today
            if student_id in self.marked:
                # Check if cooldown has passed
                last_time = self.last_seen.get(student_id, 0)
                if time.time() - last_time < self.cooldown:
                    return False
            return True
    
    def mark(self, student_id):
        """Mark student as present"""
        with self.lock:
            self.marked.add(student_id)
            self.last_seen[student_id] = time.time()
    
    def get_info(self, student_id):
        """Get student information from cache"""
        with self.lock:
            return self.info.get(student_id)
    
    def is_marked(self, student_id):
        """Check if student is already marked"""
        with self.lock:
            return student_id in self.marked
    
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            return {
                'total_loaded': len(self.encodings),
                'total_marked': len(self.marked),
                'unmarked': len(self.encodings) - len(self.marked)
            }

# Global cache instance
attendance_cache = AttendanceCache()

# ==================== FACE RECOGNITION FUNCTIONS ====================
def extract_face_encoding(image):
    """
    Extract face encoding from an image using OpenCV.
    
    Args:
        image: numpy array (BGR format from cv2)
    
    Returns:
        numpy array: Face encoding or None if no face detected
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            logger.debug("No faces detected in image")
            return None
        
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        logger.debug(f"Face detected at position ({x}, {y}) with size {w}x{h}")
        
        # Extract and resize face ROI
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (200, 200))
        
        # Create encoding using histogram + pixel data
        # 1. Histogram (256 bins)
        hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
        hist_normalized = cv2.normalize(hist, hist).flatten()
        
        # 2. Pixel data (first 1000 pixels)
        pixels = face_resized.flatten()[:1000]
        
        # Combine
        encoding = np.concatenate([hist_normalized, pixels])
        
        logger.debug(f"Face encoding created with shape {encoding.shape}")
        return encoding
        
    except Exception as e:
        logger.error(f"❌ Face encoding extraction failed: {e}")
        return None

def compare_faces(encoding1, encoding2, threshold=0.35):
    """
    Compare two face encodings using correlation coefficient.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        threshold: Matching threshold (lower = stricter)
    
    Returns:
        tuple: (is_match: bool, confidence: float)
    """
    if encoding1 is None or encoding2 is None:
        return False, 0.0
    
    try:
        # Ensure same length
        min_length = min(len(encoding1), len(encoding2))
        enc1 = encoding1[:min_length]
        enc2 = encoding2[:min_length]
        
        # Calculate correlation coefficient
        correlation_matrix = np.corrcoef(enc1, enc2)
        correlation = correlation_matrix[0, 1]
        
        # Convert to similarity score (0 to 1)
        similarity = (correlation + 1) / 2
        
        # Determine if it's a match
        is_match = similarity > (1 - threshold)
        
        logger.debug(f"Face comparison: similarity={similarity:.4f}, match={is_match}")
        return is_match, similarity
        
    except Exception as e:
        logger.error(f"❌ Face comparison failed: {e}")
        return False, 0.0

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/')
def index():
    """Home page - redirect based on login status"""
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and authentication"""
    if request.method == 'GET':
        # Already logged in
        if session.get('logged_in'):
            return redirect(url_for('dashboard'))
        return render_template('login.html')
    
    # POST - Process login
    try:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Get credentials from environment
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_password = os.getenv("ADMIN_PASSWORD", "gla123")
        
        # Validate
        if username == admin_username and password == admin_password:
            session.permanent = True
            session['logged_in'] = True
            session['username'] = username
            session['login_time'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"✅ Successful login: {username} from {request.remote_addr}")
            flash('Login successful! Welcome to Smart Attendance System.', 'success')
            return redirect(url_for('dashboard'))
        else:
            logger.warning(f"❌ Failed login attempt: username='{username}' from {request.remote_addr}")
            return render_template('login.html', error="Invalid username or password")
            
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        return render_template('login.html', error="An error occurred during login")

@app.route('/logout')
def logout():
    """Logout and clear session"""
    username = session.get('username', 'Unknown')
    session.clear()
    logger.info(f"👋 User logged out: {username}")
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# ==================== DASHBOARD ====================
@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard showing all classes with statistics"""
    try:
        # Get all classes from Firebase
        classes_ref = db.reference('classes').get() or {}
        classes_list = []
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        for class_id, class_data in classes_ref.items():
            students_dict = class_data.get('students', {})
            
            # Get today's attendance
            attendance_ref = db.reference(f'attendance/{today}/{class_id}').get() or {}
            
            present_count = len(attendance_ref)
            total_students = len(students_dict)
            
            # Calculate attendance rate
            if total_students > 0:
                attendance_rate = f"{(present_count / total_students * 100):.0f}%"
            else:
                attendance_rate = "0%"
            
            classes_list.append({
                'id': class_id,
                'name': class_data.get('name', 'Unknown Class'),
                'department': class_data.get('department', 'General'),
                'semester': class_data.get('semester', 'N/A'),
                'student_count': total_students,
                'present_today': present_count,
                'attendance_rate': attendance_rate
            })
        
        # Sort by class name
        classes_list.sort(key=lambda x: x['name'])
        
        logger.info(f"📊 Dashboard loaded: {len(classes_list)} classes displayed")
        
        return render_template('dashboard.html',
                             classes=classes_list,
                             username=session.get('username'),
                             today=today)
                             
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        flash('Error loading dashboard. Please try again.', 'error')
        return render_template('dashboard.html',
                             classes=[],
                             username=session.get('username'),
                             today=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

# ==================== CLASS MANAGEMENT ====================
@app.route('/create-class', methods=['GET', 'POST'])
@login_required
def create_class():
    """Create new class"""
    if request.method == 'GET':
        return render_template('create_class.html')
    
    # POST - Create class
    try:
        class_id = request.form.get('class_id', '').strip().upper()
        class_name = request.form.get('class_name', '').strip()
        department = request.form.get('department', 'General').strip()
        semester = request.form.get('semester', 'N/A').strip()
        
        # Validate input
        if not class_id or not class_name:
            flash('Class ID and Class Name are required!', 'error')
            return render_template('create_class.html', error="All fields are required")
        
        # Check if class ID already exists
        existing_class = db.reference(f'classes/{class_id}').get()
        if existing_class:
            flash(f'Class ID "{class_id}" already exists!', 'error')
            return render_template('create_class.html', error=f"Class ID '{class_id}' already exists")
        
        # Create class in database
        db.reference(f'classes/{class_id}').set({
            'name': class_name,
            'department': department,
            'semester': semester,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'created_by': session.get('username'),
            'students': {}
        })
        
        logger.info(f"✅ Class created: {class_id} - {class_name} by {session.get('username')}")
        flash(f'Class "{class_name}" created successfully!', 'success')
        return redirect(url_for('class_detail', class_id=class_id))
        
    except Exception as e:
        logger.error(f"❌ Create class error: {e}")
        import traceback
        traceback.print_exc()
        flash('Error creating class. Please try again.', 'error')
        return render_template('create_class.html', error=f"Error: {str(e)}")

@app.route('/class/<class_id>')
@login_required
def class_detail(class_id):
    """Class detail page showing students and attendance"""
    try:
        # Get class data
        class_data = db.reference(f'classes/{class_id}').get()
        
        if not class_data:
            logger.warning(f"⚠️ Class {class_id} not found")
            flash('Class not found!', 'error')
            return redirect(url_for('dashboard'))
        
        # Get students
        students_dict = class_data.get('students', {})
        students_list = []
        
        for student_id, student_data in students_dict.items():
            students_list.append({
                'id': student_id,
                'name': student_data.get('name', 'Unknown'),
                'roll_no': student_data.get('roll_no', 'N/A'),
                'image_url': student_data.get('image_url', ''),
                'registered_at': student_data.get('registered_at', 'N/A')
            })
        
        # Sort by roll number
        students_list.sort(key=lambda x: x['roll_no'])
        
        # Get today's attendance
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        attendance_dict = db.reference(f'attendance/{today}/{class_id}').get() or {}
        attendance_list = list(attendance_dict.keys())
        
        logger.info(f"📊 Class detail loaded: {class_id} - {len(students_list)} students, {len(attendance_list)} present today")
        
        return render_template('class_detail.html',
                             class_id=class_id,
                             class_data=class_data,
                             students=students_list,
                             attendance=attendance_list,
                             today=today)
                             
    except Exception as e:
        logger.error(f"❌ Class detail error for {class_id}: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error loading class details: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/class/<class_id>/delete', methods=['POST'])
@login_required
def delete_class(class_id):
    """API: Delete a class"""
    try:
        # Check if class exists
        class_data = db.reference(f'classes/{class_id}').get()
        if not class_data:
            return jsonify({'success': False, 'error': 'Class not found'}), 404
        
        class_name = class_data.get('name', class_id)
        
        # Delete class from database
        db.reference(f'classes/{class_id}').delete()
        
        logger.info(f"🗑️ Class deleted: {class_id} ({class_name}) by {session.get('username')}")
        return jsonify({'success': True, 'message': f'Class "{class_name}" deleted successfully'})
        
    except Exception as e:
        logger.error(f"❌ Delete class error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== STUDENT MANAGEMENT ====================
@app.route('/class/<class_id>/register')
@login_required
def register_student(class_id):
    """Student registration page with camera"""
    try:
        # Check if class exists
        class_data = db.reference(f'classes/{class_id}').get()
        
        if not class_data:
            logger.warning(f"⚠️ Class {class_id} not found")
            flash('Class not found!', 'error')
            return redirect(url_for('dashboard'))
        
        logger.info(f"📝 Student registration page opened for class: {class_id}")
        
        return render_template('register_student.html',
                             class_id=class_id,
                             class_name=class_data.get('name', 'Unknown Class'))
                             
    except Exception as e:
        logger.error(f"❌ Register page error for {class_id}: {e}")
        flash('Error loading registration page.', 'error')
        return redirect(url_for('class_detail', class_id=class_id))

@app.route('/api/student/register', methods=['POST'])
@login_required
def api_register_student():
    """API: Register a new student with face encoding"""
    try:
        data = request.json
        
        # Extract data
        class_id = data.get('class_id')
        student_id = data.get('student_id', '').strip()
        name = data.get('name', '').strip()
        roll_no = data.get('roll_no', '').strip()
        image_data = data.get('image_data')
        
        logger.info(f"📝 Registration request: {name} ({student_id}) for class {class_id}")
        
        # Validate input
        if not all([class_id, student_id, name, roll_no, image_data]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        # Check if student already exists in this class
        existing_student = db.reference(f'classes/{class_id}/students/{student_id}').get()
        if existing_student:
            logger.warning(f"⚠️ Student {student_id} already exists in class {class_id}")
            return jsonify({'success': False, 'error': f'Student ID "{student_id}" already exists in this class'}), 400
        
        # Decode image from base64
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image_pil = Image.open(BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except Exception as img_error:
            logger.error(f"❌ Image decoding error: {img_error}")
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400
        
        # Extract face encoding
        face_encoding = extract_face_encoding(image_cv)
        
        if face_encoding is None:
            logger.warning(f"⚠️ No face detected for student {student_id}")
            return jsonify({
                'success': False,
                'error': 'No face detected! Please ensure:\n• Your face is clearly visible\n• Good lighting\n• Face is centered\n• No obstructions'
            }), 400
        
        # Upload image to Firebase Storage
        image_url = ""
        try:
            blob = bucket.blob(f'students/{class_id}/{student_id}.jpg')
            img_byte_arr = BytesIO()
            image_pil.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            blob.upload_from_file(img_byte_arr, content_type='image/jpeg')
            blob.make_public()
            image_url = blob.public_url
            logger.info(f"✅ Image uploaded to Firebase Storage: {image_url}")
        except Exception as upload_error:
            logger.warning(f"⚠️ Image upload failed (continuing without image): {upload_error}")
            image_url = ""  # Continue without image URL
        
        # Save student to database
        db.reference(f'classes/{class_id}/students/{student_id}').set({
            'name': name,
            'roll_no': roll_no,
            'encoding': face_encoding.tolist(),
            'image_url': image_url,
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'registered_by': session.get('username')
        })
        
        logger.info(f"✅ Student registered successfully: {name} ({student_id}) in class {class_id}")
        return jsonify({
            'success': True,
            'message': f'Student "{name}" registered successfully!'
        })
        
    except Exception as e:
        logger.error(f"❌ Student registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Registration failed: {str(e)}'
        }), 500

@app.route('/api/student/<class_id>/<student_id>/delete', methods=['POST'])
@login_required
def delete_student(class_id, student_id):
    """API: Delete a student from class"""
    try:
        # Get student info before deletion
        student_data = db.reference(f'classes/{class_id}/students/{student_id}').get()
        
        if not student_data:
            return jsonify({'success': False, 'error': 'Student not found'}), 404
        
        student_name = student_data.get('name', student_id)
        
        # Delete from database
        db.reference(f'classes/{class_id}/students/{student_id}').delete()
        
        # Try to delete image from storage
        try:
            blob = bucket.blob(f'students/{class_id}/{student_id}.jpg')
            blob.delete()
            logger.info(f"✅ Student image deleted from storage")
        except:
            logger.warning(f"⚠️ Student image not found in storage (already deleted or never uploaded)")
        
        logger.info(f"🗑️ Student deleted: {student_name} ({student_id}) from class {class_id} by {session.get('username')}")
        return jsonify({
            'success': True,
            'message': f'Student "{student_name}" deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Delete student error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ATTENDANCE TAKING ====================
@app.route('/class/<class_id>/attendance')
@login_required
def take_attendance(class_id):
    """Take attendance page with live camera feed"""
    try:
        # Check if class exists
        class_data = db.reference(f'classes/{class_id}').get()
        
        if not class_data:
            logger.warning(f"⚠️ Class {class_id} not found")
            flash('Class not found!', 'error')
            return redirect(url_for('dashboard'))
        
        # Get statistics
        students = class_data.get('students', {})
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        attendance = db.reference(f'attendance/{today}/{class_id}').get() or {}
        
        logger.info(f"📸 Attendance page opened for class: {class_id}")
        
        return render_template('take_attendance.html',
                             class_id=class_id,
                             class_name=class_data.get('name', 'Unknown'),
                             total_students=len(students),
                             present_count=len(attendance))
                             
    except Exception as e:
        logger.error(f"❌ Take attendance page error: {e}")
        flash('Error loading attendance page.', 'error')
        return redirect(url_for('class_detail', class_id=class_id))

@app.route('/api/attendance/start/<class_id>', methods=['POST'])
@login_required
def start_attendance(class_id):
    """API: Initialize attendance session - load students into cache"""
    try:
        # Load students into cache
        loaded_count = attendance_cache.load_class(class_id)
        
        if loaded_count == 0:
            logger.warning(f"⚠️ No students found or failed to load for class {class_id}")
            return jsonify({
                'success': False,
                'error': 'No students found in this class. Please register students first.'
            }), 400
        
        logger.info(f"📸 Attendance session started for class {class_id} - {loaded_count} students loaded")
        return jsonify({
            'success': True,
            'loaded': loaded_count,
            'message': f'Session started. {loaded_count} students ready for recognition.'
        })
        
    except Exception as e:
        logger.error(f"❌ Start attendance error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/process-frame', methods=['POST'])
@login_required
def process_frame():
    """API: Process video frame for face recognition"""
    try:
        data = request.json
        class_id = data.get('class_id')
        image_data = data.get('image')
        
        if not image_data or not class_id:
            return jsonify({'success': False, 'error': 'Missing required data'}), 400
        
        # Decode image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image_pil = Image.open(BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except Exception as decode_error:
            logger.error(f"❌ Frame decode error: {decode_error}")
            return jsonify({'success': False, 'error': 'Invalid frame data'}), 400
        
        # Detect faces in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        detected_students = []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_encoding = extract_face_encoding(face_roi)
            
            if face_encoding is not None:
                # Compare with all enrolled students
                best_match_id = None
                best_confidence = 0.0
                
                for student_id, stored_encoding in attendance_cache.encodings.items():
                    is_match, confidence = compare_faces(
                        stored_encoding,
                        face_encoding,
                        threshold=0.35
                    )
                    
                    if is_match and confidence > best_confidence:
                        best_confidence = confidence
                        best_match_id = student_id
                
                # If good match found (confidence > 65%) and can be marked
                if best_match_id and best_confidence > 0.65:
                    if attendance_cache.can_mark(best_match_id):
                        attendance_cache.mark(best_match_id)
                        
                        # Get student info
                        student_info = attendance_cache.get_info(best_match_id)
                        
                        if student_info:
                            # Save attendance to database
                            db.reference(f'attendance/{today}/{class_id}/{best_match_id}').set({
                                'name': student_info['name'],
                                'roll_no': student_info['roll_no'],
                                'time': datetime.now(timezone.utc).isoformat(),
                                'status': 'present',
                                'method': 'facial_recognition',
                                'confidence': f'{best_confidence * 100:.1f}%',
                                'marked_by': 'AI System'
                            })
                            
                            detected_students.append({
                                'id': best_match_id,
                                'name': student_info['name'],
                                'roll_no': student_info['roll_no'],
                                'confidence': f'{best_confidence * 100:.1f}%'
                            })
                            
                            logger.info(f"✅ Face recognized and marked: {student_info['name']} (confidence: {best_confidence*100:.1f}%)")
        
        # Get cache stats
        cache_stats = attendance_cache.get_stats()
        
        return jsonify({
            'success': True,
            'detected': detected_students,
            'faces_found': len(faces),
            'total_marked': cache_stats['total_marked'],
            'cache_loaded': cache_stats['total_loaded']
        })
        
    except Exception as e:
        logger.error(f"❌ Process frame error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/manual', methods=['POST'])
@login_required
def mark_manual():
    """API: Manually mark student attendance"""
    try:
        data = request.json
        class_id = data.get('class_id')
        student_id = data.get('student_id')
        status = data.get('status', 'present')
        
        # Validate
        if not class_id or not student_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Get student info
        class_data = db.reference(f'classes/{class_id}').get()
        if not class_data:
            return jsonify({'success': False, 'error': 'Class not found'}), 404
        
        student = class_data.get('students', {}).get(student_id)
        if not student:
            return jsonify({'success': False, 'error': 'Student not found'}), 404
        
        # Mark attendance
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        db.reference(f'attendance/{today}/{class_id}/{student_id}').set({
            'name': student.get('name'),
            'roll_no': student.get('roll_no'),
            'time': datetime.now(timezone.utc).isoformat(),
            'status': status,
            'method': 'manual',
            'marked_by': session.get('username')
        })
        
        logger.info(f"✅ Manual attendance: {student.get('name')} marked {status} by {session.get('username')}")
        return jsonify({
            'success': True,
            'message': f"Student marked {status} successfully"
        })
        
    except Exception as e:
        logger.error(f"❌ Manual attendance error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/status/<class_id>')
@login_required
def attendance_status(class_id):
    """API: Get real-time attendance status"""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Get attendance data
        attendance_data = db.reference(f'attendance/{today}/{class_id}').get() or {}
        
        # Get class data
        class_data = db.reference(f'classes/{class_id}').get()
        if not class_data:
            return jsonify({'success': False, 'error': 'Class not found'}), 404
        
        students_dict = class_data.get('students', {})
        
        # Categorize students
        present_list = []
        absent_list = []
        
        for student_id, student_data in students_dict.items():
            student_info = {
                'id': student_id,
                'name': student_data.get('name', 'Unknown'),
                'roll_no': student_data.get('roll_no', 'N/A')
            }
            
            if student_id in attendance_data:
                att_record = attendance_data[student_id]
                student_info['time'] = att_record.get('time')
                student_info['method'] = att_record.get('method')
                student_info['confidence'] = att_record.get('confidence', 'N/A')
                present_list.append(student_info)
            else:
                absent_list.append(student_info)
        
        # Sort lists by roll number
        present_list.sort(key=lambda x: x['roll_no'])
        absent_list.sort(key=lambda x: x['roll_no'])
        
        total_students = len(students_dict)
        present_count = len(present_list)
        absent_count = len(absent_list)
        
        # Calculate attendance rate
        if total_students > 0:
            attendance_rate = f"{(present_count / total_students * 100):.0f}%"
        else:
            attendance_rate = "0%"
        
        logger.debug(f"📊 Attendance status: {present_count}/{total_students} present in class {class_id}")
        
        return jsonify({
            'success': True,
            'present': present_list,
            'absent': absent_list,
            'total': total_students,
            'present_count': present_count,
            'absent_count': absent_count,
            'attendance_rate': attendance_rate
        })
        
    except Exception as e:
        logger.error(f"❌ Attendance status error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ANALYTICS & REPORTS ====================
@app.route('/analytics/<class_id>')
@login_required
def analytics(class_id):
    """Analytics page with charts and insights"""
    try:
        # Check if class exists
        class_data = db.reference(f'classes/{class_id}').get()
        
        if not class_data:
            logger.warning(f"⚠️ Class {class_id} not found")
            flash('Class not found!', 'error')
            return redirect(url_for('dashboard'))
        
        logger.info(f"📊 Analytics page opened for class: {class_id}")
        
        return render_template('analytics.html',
                             class_id=class_id,
                             class_name=class_data.get('name', 'Unknown'))
                             
    except Exception as e:
        logger.error(f"❌ Analytics page error: {e}")
        flash('Error loading analytics.', 'error')
        return redirect(url_for('class_detail', class_id=class_id))

@app.route('/api/analytics/<class_id>')
@login_required
def api_analytics(class_id):
    """API: Get attendance analytics data"""
    try:
        # Get last 7 days attendance trend
        trend_data = []
        
        for i in range(6, -1, -1):  # 7 days ago to today
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            attendance = db.reference(f'attendance/{date}/{class_id}').get() or {}
            
            trend_data.append({
                'date': date,
                'count': len(attendance)
            })
        
        # Get class info
        class_data = db.reference(f'classes/{class_id}').get()
        total_students = len(class_data.get('students', {}))
        
        # Generate AI insights if Gemini is available
        ai_insights = None
        
        if gemini_model and total_students > 0:
            try:
                # Calculate statistics
                avg_attendance = sum(d['count'] for d in trend_data) / len(trend_data)
                avg_percentage = (avg_attendance / total_students * 100) if total_students > 0 else 0
                
                # Create prompt for Gemini
                prompt = f"""Analyze this attendance data for "{class_data.get('name', 'Class')}":

Total Students: {total_students}
7-Day Average Attendance: {avg_attendance:.1f} students ({avg_percentage:.1f}%)
Daily Trend: {json.dumps(trend_data, indent=2)}

Provide a brief analysis (maximum 150 words) covering:
1. Overall attendance pattern and trend
2. Any concerning observations or positive trends
3. One specific, actionable recommendation for improvement

Be constructive, specific, and professional."""

                response = gemini_model.generate_content(prompt)
                ai_insights = response.text
                logger.info(f"✅ AI insights generated for class {class_id}")
                
            except Exception as ai_error:
                logger.warning(f"⚠️ AI insights generation failed: {ai_error}")
                ai_insights = "AI insights temporarily unavailable. This could be due to API limits or configuration issues. Please check your Gemini API key."
        else:
            if not gemini_model:
                ai_insights = "AI insights are disabled. Please configure GEMINI_API_KEY in your .env file to enable this feature."
            elif total_students == 0:
                ai_insights = "No students registered in this class yet. AI insights will be available once students are registered and attendance data is collected."
        
        logger.info(f"📊 Analytics data generated for class {class_id}")
        
        return jsonify({
            'success': True,
            'trend': trend_data,
            'total_students': total_students,
            'insights': ai_insights
        })
        
    except Exception as e:
        logger.error(f"❌ Analytics API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.path}")
    if session.get('logged_in'):
        flash('Page not found!', 'error')
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {error}")
    flash('An internal error occurred. Please try again.', 'error')
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    logger.warning("413 error: File too large")
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

# ==================== UTILITY ROUTES ====================
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '2.0',
        'services': {
            'firebase': 'connected',
            'opencv': 'ready',
            'gemini_ai': 'enabled' if gemini_model else 'disabled'
        }
    })

# ==================== MAIN APPLICATION ENTRY POINT ====================
if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_ENV", "production") == "development"
    
    # Display startup banner
    print("\n" + "=" * 80)
    print("🎓 SMART ATTENDANCE SYSTEM - GLA UNIVERSITY".center(80))
    print("Google Solution Challenge 2026 | Team: Ctrl+Alt+Defeat".center(80))
    print("=" * 80)
    print(f"\n{'SERVICE':<20} {'STATUS':<20} {'DETAILS':<40}")
    print("-" * 80)
    print(f"{'Firebase Database':<20} {'✅ Connected':<20} {'Realtime Database Active':<40}")
    print(f"{'Firebase Storage':<20} {'✅ Connected':<20} {'Cloud Storage Active':<40}")
    print(f"{'OpenCV':<20} {'✅ Ready':<20} {'Face Detection Enabled':<40}")
    
    if gemini_model:
        print(f"{'Google Gemini AI':<20} {'✅ Enabled':<20} {'AI Insights Active':<40}")
    else:
        print(f"{'Google Gemini AI':<20} {'⚠️  Disabled':<20} {'Check GEMINI_API_KEY in .env':<40}")
    
    print("-" * 80)
    print(f"\n🌐 Server URL: http://localhost:{port}")
    print(f"👤 Admin Login: {os.getenv('ADMIN_USERNAME', 'admin')} / {os.getenv('ADMIN_PASSWORD', 'gla123')}")
    print(f"🔧 Debug Mode: {'ON' if debug_mode else 'OFF'}")
    print(f"📝 Logging: attendance_system.log")
    print("\n" + "=" * 80)
    print("Press CTRL+C to stop the server\n")
    
    # Start Flask application
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug_mode,
            threaded=True,
            use_reloader=debug_mode
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        logger.info("Server stopped by user")
    except Exception as e:
        logger.critical(f"❌ Server startup failed: {e}")
        print(f"\n❌ Failed to start server: {e}")
