from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file in SRC directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def parse_json(json_output: str):
    # First try to find JSON wrapped in markdown code blocks
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            # Extract content between ```json and ```
            json_content = "\n".join(lines[i + 1 :])
            if "```" in json_content:
                json_content = json_content.split("```")[0]
            return json_content.strip()

    # If no markdown blocks found, try to extract JSON directly
    # Look for the first '{' or '[' character
    start_idx = -1
    for i, char in enumerate(json_output):
        if char in ["{", "["]:
            start_idx = i
            break

    if start_idx == -1:
        # No JSON found, return the original string
        return json_output.strip()

    # Find the matching closing bracket
    bracket_count = 0
    is_array = json_output[start_idx] == "["
    opening_bracket = "[" if is_array else "{"
    closing_bracket = "]" if is_array else "}"

    for i in range(start_idx, len(json_output)):
        char = json_output[i]
        if char == opening_bracket:
            bracket_count += 1
        elif char == closing_bracket:
            bracket_count -= 1
            if bracket_count == 0:
                return json_output[start_idx : i + 1]

    # If we can't find proper brackets, return from start_idx to end
    return json_output[start_idx:].strip()


def extract_bounding_boxes(
    image_path: str, output_dir: str = "tools/test image/bounding_box_outputs"
) -> str:
    """Extracts bounding boxes for notable objects in an image.

    This function identifies objects in an image,
    generates their bounding box coordinates, and creates an annotated image with red boxes
    and labels. It also saves the detection data as a JSON file.

    Args:
        image_path (str): The local file path to the input image.
        output_dir (str): The directory where the annotated image and detection JSON will be saved.
                          Defaults to 'tools/test image/bounding_box_outputs'.

    Returns:
        str: The raw JSON string from the AI model containing the detection results.
    """
    # Load and resize image
    im = Image.open(image_path)
    im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    prompt = """
  Identify the objects in this image.
  Output a JSON list where each entry contains the 2D bounding box 
  in the key "box_2d" and the text label in the key "label". 
  Use descriptive labels for the identified objects.
  """
    client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    )
    model = "gemini-2.5-flash"

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        ),  # set thinking_budget to 0 for better results in object detection
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            im,
            prompt,
        ],  # Pillow images can be directly passed as inputs (which will be converted by the SDK)
        config=generate_content_config,
    )

    # Parse JSON response
    parsed_json = parse_json(response.text)
    print("Parsed JSON:")
    print(parsed_json)
    print("=" * 50)

    try:
        items = json.loads(parsed_json)
        print(
            f"Successfully parsed JSON with {len(items) if isinstance(items, list) else 'unknown'} items"
        )
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Attempted to parse: {parsed_json[:200]}...")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each bounding box
    print("Processing bounding boxes...")

    # Create a copy of the original image to draw on
    annotated_image = im.convert("RGBA")
    draw = ImageDraw.Draw(annotated_image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, item in enumerate(items):
        # Get bounding box coordinates
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * im.size[1])
        x0 = int(box[1] / 1000 * im.size[0])
        y1 = int(box[2] / 1000 * im.size[1])
        x1 = int(box[3] / 1000 * im.size[0])

        # Skip invalid boxes
        if y0 >= y1 or x0 >= x1:
            print(f"Skipping invalid box for {item['label']}: {box}")
            continue

        # Draw bounding box rectangle
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)

        # Draw label background and text
        label = item['label']
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position label above the box
        label_x = x0
        label_y = max(0, y0 - text_height - 5)
        if label_y < 0:
            label_y = y0 + 5

        # Draw label background
        draw.rectangle([label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
                      fill=(255, 0, 0), outline=(255, 0, 0))

        # Draw label text
        draw.text((label_x + 2, label_y + 2), label, fill=(255, 255, 255), font=font)

        print(f"Drew bounding box and label for {item['label']} at [{y0}, {x0}, {y1}, {x1}]")

    # Save the annotated image
    output_filename = "annotated_image.png"
    output_path = os.path.join(output_dir, output_filename)
    annotated_image.save(output_path)
    print(f"Saved annotated image to {output_path}")

    # Also save the detection results as JSON
    json_filename = "detections.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(items, f, indent=2)
    print(f"Saved detection results to {json_path}")

    return parsed_json


# Example usage
if __name__ == "__main__":
    extract_bounding_boxes("tools/test image/Screenshot 2025-08-31 at 22.13.51.png")
