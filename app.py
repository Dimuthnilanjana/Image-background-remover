from flask import Flask, render_template, request, send_file
import io
import os
from PIL import Image
import numpy as np
import cv2
from rembg import remove, new_session
import uuid
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize multiple sessions for different use cases
human_session = new_session("u2net_human_seg")  # For human subjects
general_session = new_session("isnet-general-use")  # For general objects
detail_session = new_session("u2net")  # For high detail

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded')
    
    file = request.files['image']
    if not file or file.filename == '':
        return render_template('index.html', error='No image selected')
    
    # Generate unique filenames
    temp_filename = f"upload_{uuid.uuid4()}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)
    
    try:
        # Get processing parameters
        params = {
            'alpha_matting': 'alpha_matting' in request.form,
            'foreground_threshold': int(request.form.get('foreground_threshold', 240)),
            'background_threshold': int(request.form.get('background_threshold', 10)),
            'erode_size': int(request.form.get('erode_size', 10)),
            'detail_level': request.form.get('detail_level', 'high'),
            'subject_type': request.form.get('subject_type', 'general')  # New: human/general
        }
        
        # Read image
        with open(temp_path, 'rb') as img_file:
            img_data = img_file.read()
        
        # Process image
        start_time = time.time()
        result = remove_background_enhanced(img_data, **params)
        
        # Save output
        output_filename = f"processed_{uuid.uuid4()}_{int(time.time())}.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        result.save(output_path)
        
        logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
        
        return render_template('index.html',
                            result_image=f"processed/{output_filename}",
                            original_image=f"uploads/{temp_filename}")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return render_template('index.html', error=f'Processing failed: {str(e)}')

def remove_background_enhanced(img_data, alpha_matting=True, 
                             foreground_threshold=240,
                             background_threshold=10,
                             erode_size=10,
                             detail_level='high',
                             subject_type='general'):
    """
    Enhanced background removal with multiple techniques
    
    Parameters:
    - subject_type: 'human' or 'general' to select appropriate model
    """
    # Select appropriate session based on subject type
    session = {
        'human': human_session,
        'general': general_session
    }.get(subject_type, detail_session)
    
    # Initial background removal
    output = remove(
        img_data,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=foreground_threshold,
        alpha_matting_background_threshold=background_threshold,
        alpha_matting_erode_size=erode_size
    )
    
    # Convert to numpy array
    img = Image.open(io.BytesIO(output)).convert('RGBA')
    img_array = np.array(img)
    
    if img_array.shape[2] != 4:
        return img  # Return if no alpha channel
    
    # Extract and refine alpha channel
    alpha = img_array[:,:,3]
    
    # Advanced edge refinement
    alpha_refined = refine_alpha_channel(
        alpha,
        detail_level=detail_level,
        subject_type=subject_type
    )
    
    # Apply refined alpha
    img_array[:,:,3] = alpha_refined
    
    # Additional cleanup
    if subject_type == 'human':
        img_array = cleanup_human_edges(img_array)
    
    return Image.fromarray(img_array)

def refine_alpha_channel(alpha, detail_level='high', subject_type='general'):
    """Advanced alpha channel refinement"""
    # Initial edge detection
    edges = cv2.Canny(alpha, 100, 200)
    
    if detail_level == 'low':
        return cv2.GaussianBlur(alpha, (5, 5), 0)
    
    elif detail_level == 'medium':
        # Adaptive thresholding for better edge preservation
        alpha_adaptive = cv2.adaptiveThreshold(
            alpha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.medianBlur(alpha_adaptive, 3)
    
    else:  # high detail
        # Multi-stage refinement
        # 1. Bilateral filter for edge-preserving smoothing
        alpha_smooth = cv2.bilateralFilter(alpha, 9, 75, 75)
        
        # 2. Edge enhancement with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        alpha_eroded = cv2.erode(alpha_smooth, kernel, iterations=1)
        alpha_dilated = cv2.dilate(alpha_eroded, kernel, iterations=1)
        
        # 3. Blend with original edges
        alpha_final = cv2.addWeighted(alpha_dilated, 0.7, edges, 0.3, 0)
        return np.clip(alpha_final, 0, 255)

def cleanup_human_edges(img_array):
    """Special cleanup for human subjects"""
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(img_array[:,:,:3], cv2.COLOR_RGB2HSV)
    
    # Define skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin areas
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Refine alpha channel near skin areas
    alpha = img_array[:,:,3]
    alpha[skin_mask > 0] = cv2.dilate(alpha[skin_mask > 0], None, iterations=1)
    
    return img_array

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)