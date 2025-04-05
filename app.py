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

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize multiple specialized sessions for better accuracy
human_session = new_session("u2net_human_seg")
general_session = new_session("isnet-general-use")
product_session = new_session("u2netp")  # Better for products
portrait_session = new_session("u2net_cloth_seg")  # Good for clothing

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    logger.debug("Received POST request to /remove-bg")
    
    if 'image' not in request.files:
        logger.error("No image in request.files")
        return render_template('index.html', error='No image uploaded')
    
    file = request.files['image']
    if not file or file.filename == '':
        logger.error("No file or empty filename")
        return render_template('index.html', error='No image selected')
    
    temp_filename = f"upload_{uuid.uuid4()}.png"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    logger.debug(f"Saving uploaded file to {temp_path}")
    file.save(temp_path)
    
    try:
        params = {
            'alpha_matting': 'alpha_matting' in request.form,
            'foreground_threshold': int(request.form.get('foreground_threshold', 240)),
            'background_threshold': int(request.form.get('background_threshold', 10)),
            'erode_size': int(request.form.get('erode_size', 10)),
            'subject_type': request.form.get('subject_type', 'general'),
            'fine_tune': 'fine_tune' in request.form,
            'quality': request.form.get('quality', 'high')
        }
        logger.debug(f"Processing parameters: {params}")
        
        with open(temp_path, 'rb') as img_file:
            img_data = img_file.read()
        
        start_time = time.time()
        
        # First detect subject type if auto-detection is enabled
        if params['subject_type'] == 'auto':
            params['subject_type'] = detect_subject_type(img_data)
            logger.debug(f"Auto-detected subject type: {params['subject_type']}")
        
        result = remove_background_advanced(img_data, **params)
        
        output_filename = f"processed_{uuid.uuid4()}_{int(time.time())}.png"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        logger.debug(f"Saving processed image to {output_path}")
        
        # Apply compression based on quality setting
        if params['quality'] == 'high':
            result.save(output_path, format='PNG', compress_level=1)
        else:
            result.save(output_path, format='PNG', optimize=True, compress_level=6)
        
        logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
        
        return render_template('index.html',
                            result_image=f"processed/{output_filename}",
                            original_image=f"uploads/{temp_filename}")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return render_template('index.html', error=f'Processing failed: {str(e)}')

def detect_subject_type(img_data):
    """Automatically detect subject type based on image content"""
    try:
        # Load image
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_array = np.array(img)
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Use face detection to identify humans
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            return 'human'
        
        # Check for product characteristics (high contrast, centered object)
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(gray_blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if object is centered and of reasonable size
            center_x, center_y = img_cv.shape[1] // 2, img_cv.shape[0] // 2
            contour_center_x = x + w // 2
            contour_center_y = y + h // 2
            
            is_centered = (abs(center_x - contour_center_x) < img_cv.shape[1] * 0.2 and 
                          abs(center_y - contour_center_y) < img_cv.shape[0] * 0.2)
            
            is_large_enough = (w * h) > (img_cv.shape[0] * img_cv.shape[1] * 0.1)
            
            if is_centered and is_large_enough:
                return 'product'
        
        # Default to general if nothing specific detected
        return 'general'
        
    except Exception as e:
        logger.error(f"Error in detect_subject_type: {str(e)}", exc_info=True)
        return 'general'  # Default to general on error

def remove_background_advanced(img_data, alpha_matting=True, 
                             foreground_threshold=240,
                             background_threshold=10,
                             erode_size=10,
                             subject_type='general',
                             fine_tune=False,
                             quality='high'):
    """
    Advanced background removal with multi-model approach and refinement
    """
    try:
        logger.debug(f"Starting advanced background removal for {subject_type}")
        
        # Select appropriate session based on subject type
        if subject_type == 'human':
            session = human_session
        elif subject_type == 'product':
            session = product_session
        elif subject_type == 'portrait':
            session = portrait_session
        else:
            session = general_session
        
        # Apply higher quality settings for high quality mode
        if quality == 'high':
            alpha_matting = True
            post_process_mask = True
        else:
            post_process_mask = False
        
        # Initial removal with rembg
        logger.debug("Running rembg removal")
        output = remove(
            img_data,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=foreground_threshold,
            alpha_matting_background_threshold=background_threshold,
            alpha_matting_erode_size=erode_size,
            post_process_mask=post_process_mask
        )
        
        # Load original for color correction and enhancements
        logger.debug("Loading original image")
        original_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        original_array = np.array(original_img)
        
        # Process rembg result
        logger.debug("Processing rembg output")
        img = Image.open(io.BytesIO(output)).convert('RGBA')
        img_array = np.array(img)
        
        if img_array.shape[2] != 4:
            logger.warning("Image has no alpha channel")
            return img
        
        # Extract alpha
        alpha = img_array[:,:,3]
        logger.debug(f"Alpha channel stats - min: {alpha.min()}, max: {alpha.max()}, mean: {alpha.mean():.2f}")
        
        # Apply specific enhancements based on subject type
        if subject_type == 'human':
            refined_alpha = enhance_human_mask(alpha, original_array)
        elif subject_type == 'product':
            refined_alpha = enhance_product_mask(alpha, original_array)
        elif subject_type == 'portrait':
            refined_alpha = enhance_portrait_mask(alpha, original_array)
        else:
            refined_alpha = enhance_general_mask(alpha, original_array)
        
        # Advanced post-processing for fine details
        if fine_tune:
            logger.debug("Applying fine-tuning to mask")
            refined_alpha = refine_edges(refined_alpha, original_array)
        
        # Apply refined alpha
        img_array[:,:,3] = refined_alpha
        
        # Apply color correction to maintain original colors
        logger.debug("Applying color correction")
        mask = img_array[:,:,3] > 0
        img_array[mask, :3] = original_array[mask]
        
        # Final cleanup for perfect transparency
        img_array[~mask, :3] = 0  # Zero out RGB for fully transparent areas
        
        return Image.fromarray(img_array)
    
    except Exception as e:
        logger.error(f"Error in remove_background_advanced: {str(e)}", exc_info=True)
        # Fallback to original image on error
        return Image.open(io.BytesIO(img_data))

def enhance_human_mask(alpha, original_array):
    """Enhanced mask processing for human subjects"""
    try:
        logger.debug("Enhancing human mask")
        
        # Convert to YCrCb for better skin tone detection
        ycrcb = cv2.cvtColor(original_array, cv2.COLOR_RGB2YCrCb)
        
        # Skin detection in YCrCb space (more accurate than HSV for skin)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Hair detection using grayscale and adaptive thresholding
        gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, hair_mask = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(skin_mask, hair_mask)
        
        # Create kernels for morphological operations
        kernel_small = np.ones((3,3), np.uint8)
        kernel_medium = np.ones((5,5), np.uint8)
        
        # Clean up the mask with morphological operations
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Enhanced alpha
        alpha_enhanced = np.maximum(alpha, combined_mask)
        
        # Final cleanup
        alpha_refined = cv2.GaussianBlur(alpha_enhanced, (3, 3), 0)
        alpha_refined = np.where(alpha_refined > 10, 255, 0).astype(np.uint8)
        
        return alpha_refined
    
    except Exception as e:
        logger.error(f"Error in enhance_human_mask: {str(e)}", exc_info=True)
        return alpha

def enhance_product_mask(alpha, original_array):
    """Enhanced mask processing for product images"""
    try:
        logger.debug("Enhancing product mask")
        
        # Convert to LAB color space for better color segmentation
        lab = cv2.cvtColor(original_array, cv2.COLOR_RGB2LAB)
        
        # Use L channel for luminance-based segmentation
        l_channel = lab[:,:,0]
        
        # Adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Create kernels for morphological operations
        kernel = np.ones((5,5), np.uint8)
        
        # Clean up the mask
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Combine with alpha
        alpha_enhanced = np.maximum(alpha, thresh)
        
        # Smooth edges
        alpha_refined = cv2.GaussianBlur(alpha_enhanced, (3, 3), 0)
        
        # Binarize the result with a low threshold to keep fine details
        alpha_refined = np.where(alpha_refined > 5, 255, 0).astype(np.uint8)
        
        return alpha_refined
    
    except Exception as e:
        logger.error(f"Error in enhance_product_mask: {str(e)}", exc_info=True)
        return alpha

def enhance_portrait_mask(alpha, original_array):
    """Enhanced mask processing for portrait/clothing images"""
    try:
        logger.debug("Enhancing portrait/clothing mask")
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(original_array, cv2.COLOR_RGB2HSV)
        
        # Detect high saturation regions (likely clothing)
        saturation = hsv[:,:,1]
        _, sat_mask = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
        
        # Combine with alpha
        alpha_enhanced = np.maximum(alpha, sat_mask)
        
        # Edge preservation
        edges = cv2.Canny(original_array, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Combine with edges
        alpha_enhanced = np.maximum(alpha_enhanced, dilated_edges)
        
        # Clean up
        kernel = np.ones((5,5), np.uint8)
        alpha_refined = cv2.morphologyEx(alpha_enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Binarize with threshold that preserves details
        alpha_refined = np.where(alpha_refined > 5, 255, 0).astype(np.uint8)
        
        return alpha_refined
    
    except Exception as e:
        logger.error(f"Error in enhance_portrait_mask: {str(e)}", exc_info=True)
        return alpha

def enhance_general_mask(alpha, original_array):
    """Enhanced mask processing for general objects"""
    try:
        logger.debug("Enhancing general mask")
        
        # Convert to grayscale
        gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to ensure they're included
        dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Combine with alpha
        alpha_enhanced = np.maximum(alpha, dilated_edges)
        
        # Smooth using bilateral filter to preserve edges
        alpha_float = alpha_enhanced.astype(np.float32) / 255.0
        alpha_filtered = cv2.bilateralFilter(alpha_float, 9, 75, 75)
        alpha_filtered = (alpha_filtered * 255).astype(np.uint8)
        
        # Binarize with threshold
        alpha_refined = np.where(alpha_filtered > 10, 255, 0).astype(np.uint8)
        
        return alpha_refined
    
    except Exception as e:
        logger.error(f"Error in enhance_general_mask: {str(e)}", exc_info=True)
        return alpha

def refine_edges(alpha, original_array):
    """Advanced edge refinement for fine details"""
    try:
        logger.debug("Applying edge refinement")
        
        # Convert alpha to float for processing
        alpha_float = alpha.astype(np.float32) / 255.0
        
        # Edge detection on original image
        gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create gradient mask from edges
        gradient = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        gradient = 1.0 - cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)
        
        # Guided filter for edge-aware smoothing
        guided_filter = cv2.ximgproc.createGuidedFilter(gray, 3, 0.01)
        refined = guided_filter.filter(alpha_float, gradient)
        
        # Convert back to uint8
        refined = (refined * 255).astype(np.uint8)
        
        # Final threshold to clean up
        refined = np.where(refined > 127, 255, 0).astype(np.uint8)
        
        return refined
    
    except Exception as e:
        logger.error(f"Error in refine_edges: {str(e)}", exc_info=True)
        return alpha

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)