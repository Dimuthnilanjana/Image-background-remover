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
human_session = new_session("u2net_human_seg")  # Specialized for human subjects
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
    
    temp_filename = f"upload_{uuid.uuid4()}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)
    
    try:
        params = {
            'alpha_matting': 'alpha_matting' in request.form,
            'foreground_threshold': int(request.form.get('foreground_threshold', 240)),
            'background_threshold': int(request.form.get('background_threshold', 10)),
            'erode_size': int(request.form.get('erode_size', 10)),
            'detail_level': request.form.get('detail_level', 'high'),
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
                             detail_level='high',
                             subject_type='general'):
    """
    Enhanced background removal with precise edge handling
    """
    try:
        session = {
            'human': human_session,
            'general': general_session
        }.get(subject_type, detail_session)
        
        # Initial background removal with tighter thresholds for human subjects
        output = remove(
            img_data,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=foreground_threshold,
            alpha_matting_background_threshold=background_threshold,
            alpha_matting_erode_size=erode_size,
            post_process_mask=True  # Enable rembg's built-in mask post-processing
        )
        
        # Load original image for reference
        original_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        original_array = np.array(original_img)
        
        # Process result
        img = Image.open(io.BytesIO(output)).convert('RGBA')
        img_array = np.array(img, dtype=np.uint8)
        
        if img_array.shape[2] != 4:
            logger.warning("Image has no alpha channel")
            return img
        
        # Extract alpha and enhance
        alpha = img_array[:,:,3]
        alpha_refined = refine_alpha_channel(
            alpha,
            detail_level=detail_level,
            subject_type=subject_type,
            original_array=original_array  # Pass original for reference
        )
        
        # Apply refined alpha
        img_array[:,:,3] = alpha_refined
        
        # Human-specific cleanup with improved edge handling
        if subject_type == 'human':
            img_array = cleanup_human_edges(img_array, original_array)
        
        # Final cleanup: ensure no residual background blur
        mask = img_array[:,:,3] > 0
        img_array[mask, :3] = original_array[mask]  # Use original colors where alpha exists
        
        return Image.fromarray(img_array)
    
    except Exception as e:
        logger.error(f"Error in remove_background_enhanced: {str(e)}")
        return Image.open(io.BytesIO(img_data))

def refine_alpha_channel(alpha, detail_level='high', subject_type='general', original_array=None):
    """Advanced alpha channel refinement with edge preservation"""
    try:
        # Enhance edges
        edges = cv2.Canny(alpha, 100, 200)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        if detail_level == 'low':
            refined = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        elif detail_level == 'medium':
            alpha_adaptive = cv2.adaptiveThreshold(
                alpha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            refined = cv2.medianBlur(alpha_adaptive, 3)
        
        else:  # high detail
            # Use bilateral filter for edge-preserving smoothing
            alpha_smooth = cv2.bilateralFilter(alpha, 9, 75, 75)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            alpha_eroded = cv2.erode(alpha_smooth, kernel, iterations=1)
            alpha_dilated = cv2.dilate(alpha_eroded, kernel, iterations=1)
            
            # Blend with edges for sharper boundaries
            refined = cv2.addWeighted(alpha_dilated, 0.8, edges, 0.2, 0)
            
            # Additional refinement for humans
            if subject_type == 'human' and original_array is not None:
                hsv = cv2.cvtColor(original_array, cv2.COLOR_RGB2HSV)
                hair_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))  # Dark hair detection
                refined = np.where(hair_mask > 0, cv2.dilate(refined, kernel, iterations=1), refined)
        
        # Threshold to remove residual blur
        refined = np.where(refined > 100, 255, 0).astype(np.uint8)  # Tighter threshold
        
        return refined
    
    except Exception as e:
        logger.error(f"Error in refine_alpha_channel: {str(e)}")
        return alpha

def cleanup_human_edges(img_array, original_array):
    """Precise cleanup for human subjects"""
    try:
        # Convert to HSV for skin and hair detection
        hsv = cv2.cvtColor(original_array, cv2.COLOR_RGB2HSV)
        
        # Skin detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Hair detection (expanded range for various hair colors)
        lower_hair = np.array([0, 0, 0], dtype=np.uint8)
        upper_hair = np.array([180, 255, 100], dtype=np.uint8)
        hair_mask = cv2.inRange(hsv, lower_hair, upper_hair)
        hair_mask = cv2.dilate(hair_mask, np.ones((3,3), np.uint8), iterations=2)
        
        # Combine masks
        subject_mask = cv2.bitwise_or(skin_mask, hair_mask)
        
        # Get alpha channel
        alpha = img_array[:,:,3]
        alpha_refined = alpha.copy()
        
        # Refine edges around subject
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        alpha_dilated = cv2.dilate(alpha, kernel, iterations=1)
        alpha_refined[subject_mask > 0] = 255  # Ensure full opacity for subject
        alpha_refined = cv2.bitwise_and(alpha_dilated, alpha_refined)
        
        # Remove residual background
        background_mask = cv2.bitwise_not(subject_mask)
        alpha_refined[background_mask > 0] = np.minimum(alpha_refined[background_mask > 0], alpha[background_mask > 0])
        
        # Apply refined alpha
        img_array[:,:,3] = alpha_refined
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error in cleanup_human_edges: {str(e)}")
        return img_array

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)