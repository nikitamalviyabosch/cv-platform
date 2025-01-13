# Importing all necessary libraries 
import cv2 
import os

def video_to_images(video_path, output_dir, fps=1, image_format='jpg', prefix='frame'):
    """
    Converts a video into images based on the specified frames per second (fps).

    Parameters:
    - video_path (str): Path to the input video file.
    - output_dir (str): Directory to save the extracted images.
    - fps (int): Frames per second to extract (default is 1).
    - image_format (str): Format to save the images (default is 'jpg').
    - prefix (str): Prefix for the image filenames (default is 'frame').

    Returns:
    - int: Total number of images saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Calculate frame interval

    frame_count = 0
    saved_images = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break  # Exit loop when video ends

        # Save frame if it matches the interval
        if frame_count % frame_interval == 0:
            image_name = f"{prefix}_{saved_images:05d}.{image_format}"
            image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(image_path, frame)
            saved_images += 1

        frame_count += 1

    # Release the video capture object
    video.release()

    return saved_images

# Example usage
if __name__ == "__main__":

    video_path = f"C:/Users/nvl3kor/OneDrive - Bosch Group/Work/Project/CV_Platform/Demo/demo/gaussian-splatting-3d-reconstruction/data/bicycle/bicycle.mp4"
    output_dir = f"C:/Users/nvl3kor/OneDrive - Bosch Group/Work/Project/CV_Platform/Demo/demo/gaussian-splatting-3d-reconstruction/data/bicycle/"
    fps = 6                                
    image_format = "png"
    prefix = f'{video_path.split("/")[-1].split(".")[0]}'

    total_images = video_to_images(video_path, output_dir, fps, image_format, prefix)
    print(f"Total images saved: {total_images}")