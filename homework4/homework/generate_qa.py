# Copilot was used as a teaching assistance and guide
import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

        print(f'Drew box for kart id {track_id} at ({x1_scaled}, {y1_scaled}), ({x2_scaled}, {y2_scaled})') # debug print

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    kart_objects = []

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return kart_objects

    # Image center
    img_center_x = img_width / 2
    img_center_y = img_height / 2

    # Initialize variables to track the center kart
    min_distance_to_center = float("inf")
    center_kart_object_idx = -1
    kart_object_idx = -1

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Extract kart objects
    # note: track_id is kart_id!
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Else it's a kart

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Calculate center of the kart
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        """
        # Check if the kart is within image boundaries
        if center_x < 0 or center_x > img_width or center_y < 0 or center_y > img_height:
            continue
        """

        kart_object_idx += 1

        kart_name = info["karts"][track_id]

        # Determine if this kart is closest to image center
        distance_to_center = np.sqrt((center_x - img_center_x) ** 2 + (center_y - img_center_y) ** 2)
        if distance_to_center < min_distance_to_center:
            min_distance_to_center = distance_to_center
            center_kart_object_idx = kart_object_idx

        kart_object = {
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False, # to be updated later
        }
        kart_objects.append(kart_object)

        #print(f'Found kart object: ID={track_id}, Name={kart_name}, Center=({center_x}, {center_y})') # debug print

    if center_kart_object_idx != -1:
        #print(f'Center kart object index: {center_kart_object_idx}') # debug print
        kart_objects[center_kart_object_idx]["is_center_kart"] = True

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)
    # return track name or the default value of "Unknown" if not found
    return info.get("track", "Unknown")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    question_answer_pairs = []

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # 1. Ego car question
    # What kart is the ego car?
    ego_cart = None

    for kart in kart_objects:
        if kart["is_center_kart"]:
            ego_cart = kart
            break

    ego_cart_name = ego_cart["kart_name"] if ego_cart else "Unknown"

    question_answer_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_cart_name
    })

    # 2. Total karts question
    # How many karts are there in the scenario?
    total_karts = len(kart_objects)

    question_answer_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(total_karts)
    })

    # 3. Track information questions
    # What track is this?
    question_answer_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # 4. Relative position questions for each kart
    left_right_ego = "Unknown"
    front_behind_ego = "Unknown"

    # for counting questions
    left_of_ego_count = 0
    right_of_ego_count = 0
    front_of_ego_count = 0
    behind_ego_count = 0

    for kart in kart_objects:
        if  kart == ego_cart:
                continue # Skip ego car

        # Is {kart_name} to the left or right of the ego car?
        if kart["center"][0] <= ego_cart["center"][0]:
            left_right_ego = "left"
            left_of_ego_count += 1
        else:
            left_right_ego = "right"
            right_of_ego_count += 1
    
        question_answer_pairs.append({
            "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
            "answer": left_right_ego
        })

        # Is {kart_name} in front of or behind the ego car?
        if kart["center"][1] < ego_cart["center"][1]:
            front_behind_ego = "front"
            front_of_ego_count += 1
        else:
            front_behind_ego = "back"
            behind_ego_count += 1

        #print(f'Kart {kart["kart_name"]} is {front_behind_ego} of ego car because its center y-coordinate {kart["center"][1]} is {"less than" if front_behind_ego == "front" else "greater than"} ego car center y-coordinate {ego_cart["center"][1]}') # debug print
    
        question_answer_pairs.append({
            "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
            "answer": front_behind_ego
        })

        # Where is {kart_name} relative to the ego car?
        question_answer_pairs.append({
            "question": f"Where is {kart['kart_name']} relative to the ego car?",
            "answer": f"{front_behind_ego} and {left_right_ego}"
        })

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    if left_of_ego_count != 0:
        question_answer_pairs.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(left_of_ego_count)
        })

    # How many karts are to the right of the ego car?
    if right_of_ego_count != 0:
        question_answer_pairs.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(right_of_ego_count)
        })

    # How many karts are in front of the ego car?
    if front_of_ego_count != 0:
        question_answer_pairs.append({
            "question": "How many karts are in front of the ego car?",
            "answer": str(front_of_ego_count)
        })

    # How many karts are behind the ego car?
    if behind_ego_count != 0:
        question_answer_pairs.append({
            "question": "How many karts are behind the ego car?",
            "answer": str(behind_ego_count)
        })
    return question_answer_pairs

def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def generate_qa_files(info_dir: str, output_file: str, img_width: int = 150, img_height: int = 100):
    """
    Generate QA pairs for all info.json files in the specified directory.

    Args:
        info_dir: Directory containing info.json files
        output_file: Path to the output JSON file to save QA pairs
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)
    """
    info_path = Path(info_dir)
    all_qa_pairs = []

    views_per_frame = 10
    
    # Get data subdirectory from info_path
    data_subdir = Path(info_path).name
    print(f"Generating QA pairs for info path \"{info_path}\" using data subdirectory: {data_subdir}")

    # Get list of all info files
    info_files = list(info_path.glob("*_info.json"))

    from tqdm import tqdm

    # Iterate over all info.json files in the directory
    for info_file in tqdm(info_files, desc="Processing frames"):
        #print(f"Processing {info_file}...") # debug print
        for view_index in range(views_per_frame):  # Assuming 10 views per frame
            #print(f"Processing view {view_index} of {info_file}...") # debug print
            qa_pairs = generate_qa_pairs(str(info_file), view_index, img_width, img_height)
            for qa in qa_pairs:
                qa_entry = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "image_file": f"{data_subdir}/{info_file.stem.replace('_info', '')}_{view_index:02d}_im.jpg"
                }
                all_qa_pairs.append(qa_entry)

    # Save all QA pairs to the output JSON file
    with open(output_file, "w") as f:
        json.dump(all_qa_pairs, f, indent=4)

    print(f"Generated {len(all_qa_pairs)} QA pairs and saved to {output_file}")

def main():
    fire.Fire(
        {
            "check": check_qa_pairs, 
            "generate": generate_qa_files
        }
    )

if __name__ == "__main__":
    main()
