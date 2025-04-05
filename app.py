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

# Initialize sessions - only use the most reliable models
human_session = new_session("u2net_human_seg")
general_session = new_session("u2net")  # Most reliable general model

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
    
    # Create unique filename to avoid conflicts
    temp_filename = f"upload_{uuid.uuid4()}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)
    
    try:
        # Simple parameters - focus on reliability
        params = {
            'alpha_matting': 'alpha_matting' in request.form,
            'foreground_threshold': int(request.form.get('foreground_threshold', 240)),
            'background_threshold': int(request.form.get('background_threshold', 10)),
            'erode_size': int(request.form.get('erode_size', 10)),
            'subject_type': request.form.get('subject_type', 'general')
        }
        
        with open(temp_path, 'rb') as img_file:
            img_data = img_file.read()
        
        # Process image with simplified approach
        result = process_image(img_data, params)
        
        output_filename = f"processed_{uuid.uuid4()}.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        result.save(output_path, format='PNG')
        
        return render_template('index.html',
                          result_image=f"processed/{output_filename}",
                          original_image=f"uploads/{temp_filename}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return render_template('index.html', error=f'Processing failed: {str(e)}')

def process_image(img_data, params):
    """Simple, reliable image processing"""
    try:
        # Choose the right model
        session = human_session if params['subject_type'] == 'human' else general_session
        
        # Use rembg with minimal post-processing
        output = remove(
            img_data,
            session=session,
            alpha_matting=params['alpha_matting'],
            alpha_matting_foreground_threshold=params['foreground_threshold'],
            alpha_matting_background_threshold=params['background_threshold'],
            alpha_matting_erode_size=params['erode_size']
        )
        
        # Load as PIL image and ensure RGBA
        result_img = Image.open(io.BytesIO(output)).convert('RGBA')
        
        # Basic post-processing for cleaner edges
        img_array = np.array(result_img)
        
        # Simple threshold for clean background/foreground separation
        alpha = img_array[:,:,3]
        alpha = np.where(alpha > 20, 255, 0).astype(np.uint8)
        img_array[:,:,3] = alpha
        
        # Make fully transparent pixels completely black (0,0,0,0)
        mask = alpha == 0
        img_array[mask, 0:3] = 0
        
        return Image.fromarray(img_array)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        # Return original image on failure
        return Image.open(io.BytesIO(img_data))

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)