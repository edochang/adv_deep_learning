# Copilot was used as a teaching assistance and guide
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    #print(f"Generating caption for info file: {info_path}, view index: {view_index}") # debug print

    captions = []

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Setup variables
    ego_cart = None
    num_karts = len(kart_objects)
    left_right_ego = "Unknown" # for relative position
    front_behind_ego = "Unknown" # for relative position
    kart_relative_positions = []

    for kart in kart_objects:
        if kart["is_center_kart"]:
            ego_cart = kart
            break

    for kart in kart_objects:
        kart_name = kart["kart_name"]        

        if kart["is_center_kart"]:
            ego_cart = kart
            continue # Skip ego car

        # Is {kart_name} to the left or right of the ego car?
        if kart["center"][0] <= ego_cart["center"][0]:
            left_right_ego = "left of"
        else:
            left_right_ego = "right of"

        # Is {kart_name} in front of or behind the ego car?
        if kart["center"][1] < ego_cart["center"][1]:
            front_behind_ego = "in front of"
        else:
            front_behind_ego = "behind"
    
        kart_relative_positions.append(f"{kart_name} is {front_behind_ego} the ego car.")
        kart_relative_positions.append(f"{kart_name} is {left_right_ego} the ego car.")
        position = f"{front_behind_ego} and {left_right_ego}"
        kart_relative_positions.append(f"{kart_name} is {position} the ego car.")

    # 1. Ego car
    # {kart_name} is the ego car.
    ego_cart_name = ego_cart["kart_name"] if ego_cart else "Unknown"
    captions.append(f"{ego_cart_name} is the ego car.")

    # 2. Counting
    # There are {num_karts} karts in the scenario.
    captions.append(f"There are {num_karts} karts in the scene.")

    # 3. Track name
    # The track is {track_name}.
    captions.append(f"The track is {track_name}.")

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    captions.extend(kart_relative_positions)

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def generate_captions(info_dir: str, output_file: str, img_width: int = 150, img_height: int = 100):
    info_path = Path(info_dir)
    all_captions = []

    views_per_frame = 10

    # Get data subdirectory from info_path
    data_subdir = Path(info_path).name
    print(f"Generating captions for info path \"{info_path}\" using data subdirectory: {data_subdir}")

    info_files = list(info_path.glob("*_info.json"))

    from tqdm import tqdm

    for info_file in tqdm(info_files, desc="Processing frames"):
        for view_index in range(views_per_frame):  # Assuming 10 views per frame
            captions = generate_caption(str(info_file), view_index, img_width, img_height)
            for caption in captions:
                generated_caption = {
                    "image_file": f"{data_subdir}/{info_file.stem.replace('_info', '')}_{view_index:02d}_im.jpg",
                    "caption": caption,
                }
                all_captions.append(generated_caption)

    # Save to output file
    import json

    # Dumps all captions into one file
    with open(output_file, "w") as f:
        json.dump(all_captions, f, indent=4)

    

    print(f"Generated {len(all_captions)} captions and saved to {output_file}")

def main():
    fire.Fire(
        {
            "check": check_caption,
            "generate": generate_captions
        }
    )


if __name__ == "__main__":
    main()
