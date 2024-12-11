from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from werkzeug.utils import secure_filename
from uuid import uuid4
from remover import remove_bg
from PIL import Image
from pathlib import Path

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/')
def hello_world():
    return "Hello, World!"

@app.route('/remove-background', methods=['POST'])
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
    app.run(host='0.0.0.0', port=port, debug=True)
