from flask import Flask, render_template, request, send_file
import io
import os
from PIL import Image
import numpy as np
import cv2
from rembg import remove, new_session
import uuid
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Create a specialized rembg session with advanced model
session = new_session("u2net_human_seg")  # Options: u2net, u2netp, u2net_human_seg, silueta, isnet-general-use

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    print("Form data received:", request.files)
    
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded - debug info: image field not found in request.files')
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No image selected - empty filename')
    
    # Create a temporary file to save the uploaded image
    temp_filename = f"upload_{uuid.uuid4()}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    
    # Save the uploaded file
    file.save(temp_path)
    
    try:
        # Get post-processing parameters
        alpha_matting = 'alpha_matting' in request.form
        alpha_matting_foreground_threshold = int(request.form.get('foreground_threshold', 240))
        alpha_matting_background_threshold = int(request.form.get('background_threshold', 10))
        alpha_matting_erode_size = int(request.form.get('erode_size', 10))
        detail_level = request.form.get('detail_level', 'high')
        
        # Read image from saved file
        with open(temp_path, 'rb') as img_file:
            img_data = img_file.read()
        
        # Process image with enhanced technique
        result = remove_background_enhanced(
            img_data, 
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            detail_level=detail_level
        )
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        timestamp = int(time.time())
        output_filename = f"{unique_id}_{timestamp}.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Save processed image
        result.save(output_path)
        
        # Return result
        return render_template('index.html', 
                              result_image=f"processed/{output_filename}",
                              original_image=f"uploads/{temp_filename}")
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error=f'Error processing image: {str(e)}')

def remove_background_enhanced(img_data, alpha_matting=True, 
                              alpha_matting_foreground_threshold=240,
                              alpha_matting_background_threshold=10,
                              alpha_matting_erode_size=10,
                              detail_level='high'):
    """
    Enhanced background removal with advanced techniques for better details
    
    Parameters:
    - img_data: Raw image data
    - alpha_matting: Whether to use alpha matting for better edge handling
    - alpha_matting_foreground_threshold: Alpha matting foreground threshold (higher = more foreground)
    - alpha_matting_background_threshold: Alpha matting background threshold (lower = less background)
    - alpha_matting_erode_size: Erosion size for alpha matting
    - detail_level: Level of detail to preserve (low, medium, high)
    """
    # First pass with rembg using selected model
    output = remove(
        img_data,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size
    )
    
    # Convert to numpy array for additional processing
    output_img = Image.open(io.BytesIO(output))
    img_array = np.array(output_img)
    
    # Additional processing based on detail level
    if img_array.shape[2] == 4:  # Check if alpha channel exists
        # Extract alpha channel
        alpha = img_array[:,:,3]
        
        # Apply different levels of processing based on detail level
        if detail_level == 'low':
            # Smooth edges but lose some detail
            refined_alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            
        elif detail_level == 'medium':
            # Balanced approach
            kernel = np.ones((3,3), np.uint8)
            alpha_eroded = cv2.erode(alpha, kernel, iterations=1)
            alpha_dilated = cv2.dilate(alpha_eroded, kernel, iterations=1)
            refined_alpha = cv2.GaussianBlur(alpha_dilated, (3, 3), 0)
            
        else:  # high detail (default)
            # Enhanced edge preservation
            # First apply bilateral filter to smooth while preserving edges
            refined_alpha = cv2.bilateralFilter(alpha, 9, 75, 75)
            
            # Edge enhancement
            kernel_sharpen = np.array([[-1,-1,-1],
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
            refined_alpha = cv2.filter2D(refined_alpha, -1, kernel_sharpen)
            
            # Normalize to ensure values are in proper range
            refined_alpha = np.clip(refined_alpha, 0, 255)
        
        # Replace original alpha channel with refined one
        img_array[:,:,3] = refined_alpha
    
    # Convert back to PIL Image
    result_img = Image.fromarray(img_array)
    return result_img

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)