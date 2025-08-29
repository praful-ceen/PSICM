import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy import ndimage
import glob
import os


def load_focus_stack(image_paths):
    """Load all images in the focus stack"""
    images = []

    for path in sorted(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load {path}")
            continue
        images.append(img)
        print(f"Loaded: {path}")

    return images


def calculate_focus_measure_laplacian(image):
    """Calculate focus measure using Laplacian variance"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F)

    # Return absolute values for focus measure
    return np.abs(laplacian)


def calculate_focus_measure_gradient(image):
    """Calculate focus measure using gradient magnitude"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Calculate gradients
    grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient_magnitude


def focus_stack_simple(images, method='laplacian', window_size=15):
    """
    Simple focus stacking using local focus measures

    Parameters:
    - images: list of images in the focus stack
    - method: 'laplacian' or 'gradient' for focus measure
    - window_size: size of local window for focus comparison
    """
    if not images:
        raise ValueError("No images provided")

    # Ensure all images have the same size
    height, width = images[0].shape[:2]
    for i, img in enumerate(images):
        if img.shape[:2] != (height, width):
            images[i] = cv2.resize(img, (width, height))

    print(f"Processing {len(images)} images of size {width}x{height}")

    # Calculate focus measures for all images
    focus_measures = []
    for i, img in enumerate(images):
        print(f"Calculating focus measure for image {i + 1}/{len(images)}")

        if method == 'laplacian':
            focus_measure = calculate_focus_measure_laplacian(img)
        else:  # gradient
            focus_measure = calculate_focus_measure_gradient(img)

        focus_measures.append(focus_measure)

    # Convert to numpy array for easier processing
    focus_stack = np.array(focus_measures)  # Shape: (num_images, height, width)
    image_stack = np.array(images)  # Shape: (num_images, height, width, channels)

    print("Creating composite image...")

    # Initialize output image
    if len(images[0].shape) == 3:
        output = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        output = np.zeros((height, width), dtype=np.uint8)

    # Create a map showing which image was selected for each pixel
    selection_map = np.zeros((height, width), dtype=np.uint8)

    # Process in blocks for efficiency and better local focus detection
    half_window = window_size // 2

    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            # Define window boundaries
            y1 = max(0, y)
            y2 = min(height, y + window_size)
            x1 = max(0, x)
            x2 = min(width, x + window_size)

            # Get focus measures for this window from all images
            window_focus = focus_stack[:, y1:y2, x1:x2]

            # Calculate mean focus measure for each image in this window
            mean_focus = np.mean(window_focus, axis=(1, 2))

            # Select the image with highest focus for this window
            best_image_idx = np.argmax(mean_focus)

            # Copy pixels from the best focused image
            if len(images[0].shape) == 3:
                output[y1:y2, x1:x2] = image_stack[best_image_idx, y1:y2, x1:x2]
            else:
                output[y1:y2, x1:x2] = image_stack[best_image_idx, y1:y2, x1:x2]

            # Mark selection in selection map
            selection_map[y1:y2, x1:x2] = best_image_idx

    return output, selection_map


def focus_stack_advanced(images, method='laplacian'):
    """
    Advanced focus stacking using pixel-level focus comparison with smoothing
    """
    if not images:
        raise ValueError("No images provided")

    # Ensure all images have the same size
    height, width = images[0].shape[:2]
    for i, img in enumerate(images):
        if img.shape[:2] != (height, width):
            images[i] = cv2.resize(img, (width, height))

    print(f"Advanced processing {len(images)} images of size {width}x{height}")

    # Calculate focus measures for all images
    focus_measures = []
    for i, img in enumerate(images):
        print(f"Calculating focus measure for image {i + 1}/{len(images)}")

        if method == 'laplacian':
            focus_measure = calculate_focus_measure_laplacian(img)
        else:
            focus_measure = calculate_focus_measure_gradient(img)

        # Apply Gaussian smoothing to focus measure to reduce noise
        focus_measure_smooth = cv2.GaussianBlur(focus_measure, (5, 5), 1.0)
        focus_measures.append(focus_measure_smooth)

    # Convert to numpy array
    focus_stack = np.array(focus_measures)
    image_stack = np.array(images)

    print("Creating pixel-wise composite...")

    # Find best focused image for each pixel
    best_focus_indices = np.argmax(focus_stack, axis=0)

    # Initialize output
    if len(images[0].shape) == 3:
        output = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            for i in range(len(images)):
                mask = (best_focus_indices == i)
                output[mask, c] = image_stack[i, mask, c]
    else:
        output = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(images)):
            mask = (best_focus_indices == i)
            output[mask] = image_stack[i, mask]

    return output, best_focus_indices


def visualize_focus_stack_results(images, output, selection_map):
    """Visualize the focus stacking results"""
    num_images = len(images)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Show some original images
    for i in range(min(6, num_images)):
        row = i // 3
        col = i % 3
        if len(images[i].shape) == 3:
            axes[row, col].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            axes[row, col].imshow(images[i], cmap='gray')
        axes[row, col].set_title(f'Original Image {i + 1}')
        axes[row, col].axis('off')

    # Fill remaining spots in first two rows if needed
    for i in range(num_images, 6):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')

    # Show composite result
    if len(output.shape) == 3:
        axes[2, 0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    else:
        axes[2, 0].imshow(output, cmap='gray')
    axes[2, 0].set_title('Focus Stacked Result')
    axes[2, 0].axis('off')

    # Show selection map
    axes[2, 1].imshow(selection_map, cmap='tab10')
    axes[2, 1].set_title('Selection Map\n(Colors show which image was used)')
    axes[2, 1].axis('off')

    # Show focus quality visualization
    if len(output.shape) == 3:
        gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    else:
        gray_output = output

    focus_quality = calculate_focus_measure_laplacian(gray_output)
    axes[2, 2].imshow(focus_quality, cmap='hot')
    axes[2, 2].set_title('Focus Quality Map')
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.show()


def focus_stack_from_folder(folder_path, output_name="focus_stacked_result.jpg", method='advanced'):
    """
    Main function to perform focus stacking on images in a folder

    Parameters:
    - folder_path: path to folder containing focus stack images
    - output_name: name for output image
    - method: 'simple' or 'advanced'
    """

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if not image_paths:
        raise ValueError(f"No images found in {folder_path}")

    print(f"Found {len(image_paths)} images")

    # Load images
    images = load_focus_stack(image_paths)

    if len(images) < 2:
        raise ValueError("Need at least 2 images for focus stacking")

    # Perform focus stacking
    if method == 'simple':
        print("Using simple focus stacking method...")
        output, selection_map = focus_stack_simple(images, method='laplacian', window_size=20)
    else:
        print("Using advanced focus stacking method...")
        output, selection_map = focus_stack_advanced(images, method='laplacian')

    # Save result
    cv2.imwrite(output_name, output)
    print(f"Focus stacked image saved as: {output_name}")

    # Visualize results
    visualize_focus_stack_results(images, output, selection_map)

    return output, selection_map


# Example usage
if __name__ == "__main__":
    # Method 1: Process images from a folder
    try:
        folder_path = "collection"  # Change this to your folder path
        output, selection_map = focus_stack_from_folder(
            folder_path,
            "my_focus_stacked_result.jpg",
            method='advanced'  # or 'simple'
        )

    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this script:")
        print("1. Put all 7 images in a folder")
        print("2. Update the folder_path variable above")
        print("3. Run the script")

    # Method 2: Process specific image files
    """
    image_files = [
        "image_focus_1.jpg",
        "image_focus_2.jpg", 
        "image_focus_3.jpg",
        "image_focus_4.jpg",
        "image_focus_5.jpg",
        "image_focus_6.jpg",
        "image_focus_7.jpg"
    ]

    images = load_focus_stack(image_files)
    output, selection_map = focus_stack_advanced(images)
    cv2.imwrite("focus_stacked_result.jpg", output)
    visualize_focus_stack_results(images, output, selection_map)
    """