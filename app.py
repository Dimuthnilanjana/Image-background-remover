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

# Initialize sessions
human_session = new_session("u2net_human_seg")  # For human subjects
general_session = new_session("isnet-general-use")  # For general objects

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
    
    temp_filename = f"upload_{uuid.uuid4()}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)
    
    try:
        params = {
            'alpha_matting': 'alpha_matting' in request.form,
            'foreground_threshold': int(request.form.get('foreground_threshold', 240)),
            'background_threshold': int(request.form.get('background_threshold', 10)),
            'erode_size': int(request.form.get('erode_size', 10)),
            'subject_type': request.form.get('subject_type', 'general')
        }
        
        with open(temp_path, 'rb') as img_file:
            img_data = img_file.read()
        
        start_time = time.time()
        result = remove_background_enhanced(img_data, **params)
        
        output_filename = f"processed_{uuid.uuid4()}_{int(time.time())}.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        result.save(output_path, format='PNG', optimize=True)
        
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
                             subject_type='general'):
    """
    Straightforward, effective background removal
    """
    try:
        # Select session
        session = human_session if subject_type == 'human' else general_session
        
        # Perform background removal
        output = remove(
            img_data,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=foreground_threshold,
            alpha_matting_background_threshold=background_threshold,
            alpha_matting_erode_size=erode_size,
            post_process_mask=True
        )
        
        # Load original for color preservation
        original_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        original_array = np.array(original_img, dtype=np.uint8)
        
        # Process result
        img = Image.open(io.BytesIO(output)).convert('RGBA')
        img_array = np.array(img, dtype=np.uint8)
        
        if img_array.shape[2] != 4:
            logger.warning("Image has no alpha channel")
            return img
        
        # Get alpha channel
        alpha = img_array[:,:,3]
        
        # Clean up alpha: trust rembg's output, just threshold it
        alpha_refined = np.where(alpha > 30, 255, 0).astype(np.uint8)  # Low threshold to catch all foreground
        
        # Human-specific cleanup
        if subject_type == 'human':
            alpha_refined = cleanup_human_edges(alpha_refined, original_array)
        
        # Apply alpha
        img_array[:,:,3] = alpha_refined
        
        # Preserve original colors, clear background
        mask = img_array[:,:,3] > 0
        img_array[mask, :3] = original_array[mask]
        img_array[~mask, :3] = 0
        
        return Image.fromarray(img_array)
    
    except Exception as e:
        logger.error(f"Error in remove_background_enhanced: {str(e)}")
        return Image.open(io.BytesIO(img_data))

def cleanup_human_edges(alpha, original_array):
    """Minimal, effective cleanup for human subjects"""
    try:
        # Convert to HSV for subject detection
        hsv = cv2.cvtColor(original_array, cv2.COLOR_RGB2HSV)
        
        # Broad subject detection (skin, hair, clothing)
        lower_bound = np.array([0, 5, 20], dtype=np.uint8)  # Very permissive
        upper_bound = np.array([180, 255, 255], dtype=np.uint8)
        subject_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Expand subject area slightly
        kernel = np.ones((5,5), np.uint8)
        subject_mask = cv2.dilate(subject_mask, kernel, iterations=1)
        
        # Ensure subject is fully included
        alpha_refined = alpha.copy()
        alpha_refined[subject_mask > 0] = 255
        
        # Clean residual background
        background_mask = cv2.bitwise_not(subject_mask)
        alpha_refined[background_mask > 0] = np.where(alpha[background_mask > 0] < 50, 0, alpha[background_mask > 0])
        
        return alpha_refined
    
    except Exception as e:
        logger.error(f"Error in cleanup_human_edges: {str(e)}")
        return alpha

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)