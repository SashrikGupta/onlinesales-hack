from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from werkzeug.utils import secure_filename
from uuid import uuid4
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch 
import cv2 
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    # orig_im_size=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

def plot_rect(image, x_min, y_min, width, height, ax):

    
    
    ax.imshow(image)
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x_min, y_min - 5, "Label", color="red", fontsize=10)

    ax.axis('off')


def run_pipeline(image, x_min, y_min, width, height, total_run):
    # Read the image
    image = image.convert("RGB") 
    image_array = np.array(image)
    
    # Set up the white background image
    white_background = np.ones_like(image_array, dtype=np.uint8) * 255
    white_background[y_min:y_min+height, x_min:x_min+width] = image_array[y_min:y_min+height, x_min:x_min+width]
    cropped_image_region = image_array[y_min:y_min+height, x_min:x_min+width]

    orig_im = np.array(cropped_image_region)
    orig_im_size = orig_im.shape[0:2]
    model_input_size = orig_im_size

    # Set the initial image to be processed
    current_image = preprocess_image(orig_im, model_input_size).to(device)
    
    for run in range(total_run):
        # Run the model
        result = model(current_image)
        result_image = postprocess_image(result[0][0], orig_im_size)
        mask = result_image

        # Apply the mask to the image
        final_img = np.zeros_like(orig_im)
        for i, row in enumerate(orig_im): 
            for j, pixel in enumerate(row): 
                if (mask[i][j] / 255 > 0.8):
                    filter = mask[i][j] / 255
                    final_img[i][j][0] = np.ceil(pixel[0] * filter) 
                    final_img[i][j][1] = np.ceil(pixel[1] * filter)
                    final_img[i][j][2] = np.ceil(pixel[2] * filter)
                else: 
                    final_img[i][j][0] = 255 
                    final_img[i][j][1] = 255 
                    final_img[i][j][2] = 255 
        
        # Update the current image for the next iteration
        # Place the processed final image back into the region of interest on the white background
        white_image = np.ones_like(image_array, dtype=np.uint8) * 255
        white_image[y_min:y_min+height, x_min:x_min+width] = final_img
        final_image = Image.fromarray(white_image)

        # Update the image to be used in the next iteration
        # Crop the processed region again and prepare it for the next pass
        cropped_image_region = final_img
        orig_im = np.array(cropped_image_region)
        current_image = preprocess_image(orig_im, model_input_size).to(device)
    
    # Return the final processed image after all iterations
    return final_image


from PIL import Image
import numpy as np
from skimage import io

def resize_image_and_adjust_bbox(image_path,x_min, y_min, width, height ,  target_width=400, target_height=300):

    image = np.array(Image.open(image_path))
    original_height, original_width = image.shape[:2]

    resized_image = Image.fromarray(image)
    resized_image = resized_image.resize((target_width, target_height), Image.LANCZOS)
    resized_image_array = np.array(resized_image)
 
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    x_min_resized = int(x_min * scale_x)
    y_min_resized = int(y_min * scale_y)
    width_resized = int(width * scale_x)
    height_resized = int(height * scale_y)

    return resized_image_array, (x_min_resized, y_min_resized, width_resized, height_resized) , (original_width , original_height)


def resize_to_original(image, x_min, y_min, width, height, original_width, original_height, target_width=400, target_height=300):
    # Ensure 'image' is a valid image array (not a tuple)
    if isinstance(image, np.ndarray):
        resized_back_image = Image.fromarray(image).resize((original_width, original_height), Image.LANCZOS)
    else:
        raise ValueError("Provided image is not in a valid format")

    resized_back_image_array = np.array(resized_back_image)

    # Calculate scaling factors
    scale_x = original_width / target_width
    scale_y = original_height / target_height

    # Adjust bounding box coordinates and size back to the original scale
    x_min_original = int(x_min * scale_x)
    y_min_original = int(y_min * scale_y)
    width_original = int(width * scale_x)
    height_original = int(height * scale_y)

    return resized_back_image_array, (x_min_original, y_min_original, width_original, height_original)
def convert_to_rgb(image_path):
    image = Image.open(image_path)
    return image.convert("RGB")
def smooth_and_sharpen(image):
    """
    Apply smoothing to the image using Gaussian blur and then sharpen it using Laplacian filter.

    :param image: Input image (numpy array)
    :return: Processed image with smoothing and sharpening applied
    """
    # Apply Gaussian blur for smoothing
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert to grayscale for Laplacian filter
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    else:
        gray = smoothed

    # Apply Laplacian filter for sharpening
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Convert Laplacian result back to RGB if original image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        laplacian_rgb = cv2.cvtColor(cv2.convertScaleAbs(laplacian), cv2.COLOR_GRAY2RGB)
    else:
        laplacian_rgb = cv2.convertScaleAbs(laplacian)

    # Add the Laplacian result to the smoothed image for sharpening
    sharpened = cv2.addWeighted(smoothed, 1, laplacian_rgb, -1, 0)

    return sharpened

def show_pipeline(image_path, x_min, y_min, width, height):
    # Process image for resizing and bounding box adjustment
    image, (x_min, y_min, width, height), (ow, oh) = resize_image_and_adjust_bbox(image_path, x_min, y_min, width, height)
    image = Image.fromarray(image)
    
    # Apply the pipeline (ensure input is RGB)
    final = run_pipeline(image, x_min, y_min, width, height, 2)
    
    # Resize back to original dimensions
    final_image, (x_min_orig, y_min_orig, width_orig, height_orig) = resize_to_original(
        np.array(final), x_min, y_min, width, height, ow, oh
    )
    
    return final_image


import cloudinary
import cloudinary.uploader
from PIL import Image
import os 
# Configure Cloudinary with your credentials
cloudinary.config( 
    cloud_name = "dogfmhpfc", 
    api_key = "187934411863936", 
    api_secret = "hNSE54C0jo1crzIQTyEBF7uRz5g", # Click 'View API Keys' above to copy your API secret
    secure=True
)





def convert_and_upload(image_array, output_path="output.png"):
    """
    Converts an image (numpy array) to PNG format with transparency, saves it locally, and uploads it to Cloudinary.

    Args:
        image_array (np.array): The image as a numpy array.
        output_path (str): Local path to save the converted PNG image (default: "output.png").

    Returns:
        str: URL of the uploaded image on Cloudinary.
    """
    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_array)
    
    # Create an RGBA image with transparency
    rgba_image = image.convert("RGBA")
    data = rgba_image.getdata()
    new_data = []
    
    for item in data:
        # If the pixel is white (255, 255, 255), make it fully transparent
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))  # Transparent
        else:
            new_data.append(item)  # Keep the pixel as is

    rgba_image.putdata(new_data)

    # Save the image as PNG with transparency
    rgba_image.save(output_path, "PNG")

    # Upload the image to Cloudinary
    response = cloudinary.uploader.upload(output_path)

    os.remove(output_path)

    # Return the URL of the uploaded image
    return response.get("secure_url")

def remove_bg(image_path, x_min, y_min, width, height):
    """
    Removes the background from the image and uploads the transparent PNG file to Cloudinary.

    Args:
        image_path (str): Path to the input image.
        x_min, y_min, width, height: Bounding box coordinates for the object.

    Returns:
        str: URL of the uploaded PNG file with transparency.
    """
    # Process and remove background
    final = show_pipeline(image_path, x_min, y_min, width, height)
    
    # Convert to PNG with transparency and upload
    image_url = convert_and_upload(final)
    return image_url

























flask_app = Flask(__name__)

# Enable CORS for all routes
CORS(flask_app)

@flask_app.route('/')
def hello_world():
    return "Hello, World!"

@flask_app.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        # Parse JSON input
        data = request.json
        image_url = data.get("image_url")
        bounding_box = data.get("bounding_box", {})
        
        if not image_url or not bounding_box:
            return jsonify({"error": "Invalid input. Image URL and bounding box are required."}), 400
        
        x_min = bounding_box.get("x_min")
        y_min = bounding_box.get("y_min")
        x_max = bounding_box.get("x_max")
        y_max = bounding_box.get("y_max")
        
        if None in (x_min, y_min, x_max, y_max):
            return jsonify({"error": "Invalid bounding box coordinates."}), 400
        
        # Download the image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download the image from the URL."}), 400
        
        temp_dir = Path("/tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid4()}.png"
        local_image_path = temp_dir / filename
        with open(local_image_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        # Convert to .jpg if needed
        if not str(local_image_path).endswith(".jpg"):
            converted_path = local_image_path.with_suffix(".jpg")
            Image.open(local_image_path).convert("RGB").save(converted_path, "JPEG")
            os.remove(local_image_path)
            local_image_path = converted_path

        # Convert bounding box coordinates
        width = x_max - x_min
        height = y_max - y_min

        # Call the `remove_bg` function
        processed_image_url = remove_bg(str(local_image_path), x_min, y_min, width, height)

        # Remove the downloaded image
        os.remove(local_image_path)

        # Return the response
        return jsonify({
            "original_image_url": image_url,
            "processed_image_url": processed_image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    flask_app.run(host='0.0.0.0', port=port, debug=True)
    

