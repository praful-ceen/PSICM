import cv2
import numpy as np
from skimage import morphology, measure
from skimage.morphology import skeletonize, remove_small_objects
import matplotlib.pyplot as plt
from scipy import ndimage


def load_and_preprocess(image_path):
    """Load and preprocess the fluorescence microscopy image"""
    img = cv2.imread(image_path)

    if len(img.shape) == 3:
        gray = img[:, :, 1]  # Extract green channel
    else:
        gray = img

    return img, gray


def enhance_junctions(gray_img):
    """Enhance cell boundaries using CLAHE and bilateral filtering"""
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)

    # More aggressive bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(enhanced, d=11, sigmaColor=50, sigmaSpace=50)

    return bilateral


def create_clean_skeleton(enhanced_img):
    """Create a clean skeleton with minimal gaps and spurs"""
    # Threshold to get binary image
    _, binary = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

    # More aggressive gap filling to close skeleton breaks
    # First pass with large kernel
    kernel_close_large = np.ones((9, 9), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_large, iterations=3)

    # Second pass with medium kernel for fine gaps
    kernel_close_medium = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_medium, iterations=2)

    # Remove small noise
    kernel_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Remove small objects
    binary_clean = remove_small_objects(binary > 0, min_size=150, connectivity=2)

    # Additional dilation to ensure connectivity before skeletonization
    kernel_dilate = np.ones((3, 3), np.uint8)
    binary_clean = cv2.dilate(binary_clean.astype(np.uint8), kernel_dilate, iterations=1)

    # Skeletonize
    skeleton = skeletonize(binary_clean > 0)

    # Aggressive spur removal
    skeleton_clean = remove_spurs_iteratively(skeleton, max_spur_length=20, iterations=4)

    return skeleton_clean


def remove_spurs_iteratively(skeleton, max_spur_length=15, iterations=3):
    """Remove spurs (short branches) iteratively"""
    cleaned = skeleton.copy()

    for iteration in range(iterations):
        # Find endpoints
        endpoints = find_endpoints(cleaned)

        # Remove spurs from each endpoint
        for y, x in zip(*np.where(endpoints)):
            spur_pixels = trace_spur(cleaned, y, x, max_spur_length)
            if len(spur_pixels) <= max_spur_length:
                # Remove this spur
                for sy, sx in spur_pixels:
                    cleaned[sy, sx] = False

        # Remove very small connected components
        cleaned = remove_small_objects(cleaned, min_size=max_spur_length, connectivity=2)

    return cleaned


def find_endpoints(skeleton):
    """Find endpoints (pixels with exactly 1 neighbor)"""
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = (neighbor_count == 1) & skeleton
    return endpoints


def trace_spur(skeleton, start_y, start_x, max_length):
    """Trace a spur from an endpoint"""
    spur_pixels = [(start_y, start_x)]
    visited = np.zeros_like(skeleton, dtype=bool)
    visited[start_y, start_x] = True

    current_y, current_x = start_y, start_x

    for _ in range(max_length):
        # Find unvisited neighbors
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = current_y + dy, current_x + dx
                if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] and not visited[ny, nx]):
                    neighbors.append((ny, nx))

        if len(neighbors) == 0:
            # Dead end
            break
        elif len(neighbors) == 1:
            # Continue along spur
            ny, nx = neighbors[0]
            spur_pixels.append((ny, nx))
            visited[ny, nx] = True
            current_y, current_x = ny, nx
        else:
            # Hit a junction - stop here
            break

    return spur_pixels


def classify_skeleton_pixels(skeleton):
    """Classify skeleton pixels based on number of neighbors"""
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0

    # Count neighbors for each pixel
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    neighbor_count = neighbor_count * skeleton

    # Classify pixels
    endpoints = (neighbor_count == 1) & skeleton
    edge_points = (neighbor_count == 2) & skeleton
    junction_points = (neighbor_count >= 3) & skeleton

    return junction_points, edge_points, endpoints, neighbor_count


def remove_border_endpoints(skeleton, border_margin=5):
    """Remove endpoints that are close to image borders"""
    h, w = skeleton.shape
    cleaned = skeleton.copy()

    endpoints = find_endpoints(skeleton)
    endpoint_coords = np.where(endpoints)

    for y, x in zip(endpoint_coords[0], endpoint_coords[1]):
        # Check if endpoint is near border
        if (y < border_margin or y >= h - border_margin or
                x < border_margin or x >= w - border_margin):
            # Remove this endpoint and trace back to remove the branch
            branch_pixels = trace_spur(skeleton, y, x, max_length=20)
            for by, bx in branch_pixels:
                cleaned[by, bx] = False

    return cleaned


def remove_junction_endpoint_branches(skeleton):
    """Remove branches that connect junction points directly to endpoints if less than 40 pixels"""
    cleaned = skeleton.copy()

    # Iterate until no more changes (some branches may be revealed after others are removed)
    max_iterations = 5
    for iteration in range(max_iterations):
        junction_points, _, endpoints, _ = classify_skeleton_pixels(cleaned)

        # Find all endpoints
        endpoint_coords = np.where(endpoints)
        branches_removed = False

        for ey, ex in zip(endpoint_coords[0], endpoint_coords[1]):
            # Trace from endpoint to see if it leads directly to a junction
            branch_pixels = trace_branch_to_junction(cleaned, ey, ex, junction_points)

            # If branch is short (less than 20 pixels) and leads to junction, remove it
            if branch_pixels and len(branch_pixels) < 40:
                # Remove the entire branch except the junction point
                for by, bx in branch_pixels[:-1]:  # Keep the junction point
                    cleaned[by, bx] = False
                branches_removed = True

        # If no branches were removed this iteration, we're done
        if not branches_removed:
            break

    return cleaned


def trace_branch_to_junction(skeleton, start_y, start_x, junction_points):
    """Trace from endpoint to junction point"""
    branch_pixels = [(start_y, start_x)]
    visited = np.zeros_like(skeleton, dtype=bool)
    visited[start_y, start_x] = True

    current_y, current_x = start_y, start_x

    for _ in range(50):  # Max trace length
        # Find unvisited neighbors
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = current_y + dy, current_x + dx
                if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] and not visited[ny, nx]):
                    neighbors.append((ny, nx))

        if len(neighbors) == 0:
            # Dead end
            return None
        elif len(neighbors) == 1:
            # Continue along branch
            ny, nx = neighbors[0]
            branch_pixels.append((ny, nx))
            visited[ny, nx] = True

            # Check if we reached a junction
            if junction_points[ny, nx]:
                return branch_pixels

            current_y, current_x = ny, nx
        else:
            # Hit a junction
            return branch_pixels

    return None


def count_actual_junction_points(junction_points):
    """Count actual junction points by clustering nearby junction pixels"""
    if not np.any(junction_points):
        return 0, np.zeros_like(junction_points, dtype=bool)

    # Label connected components of junction pixels
    labeled_junctions = measure.label(junction_points, connectivity=2)
    num_junction_clusters = labeled_junctions.max()

    # Create a mask with one representative point per cluster
    junction_representatives = np.zeros_like(junction_points, dtype=bool)

    for i in range(1, num_junction_clusters + 1):
        # Find all pixels in this cluster
        cluster_mask = labeled_junctions == i
        cluster_coords = np.where(cluster_mask)

        if len(cluster_coords[0]) > 0:
            # Use the centroid of the cluster as representative
            center_y = int(np.mean(cluster_coords[0]))
            center_x = int(np.mean(cluster_coords[1]))
            junction_representatives[center_y, center_x] = True

    return num_junction_clusters, junction_representatives


def count_edge_segments(skeleton, junction_points):
    """Count distinct edge segments (bicellular junctions)"""
    # Remove junction points to isolate edge segments
    edges_only = skeleton & ~junction_points

    # Label connected components
    labeled_edges = measure.label(edges_only, connectivity=2)
    num_edge_segments = labeled_edges.max()

    return num_edge_segments, labeled_edges


def visualize_results(img, skeleton, junction_points, edge_points, endpoints):
    """Visualize the detected junctions and skeleton"""
    if len(img.shape) == 2:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        output = img.copy()

    # Create overlay
    overlay = np.zeros_like(output)

    # Draw skeleton in white
    overlay[skeleton > 0] = [255, 255, 255]

    # Draw junction points in red (larger circles)
    junction_coords = np.where(junction_points)
    for y, x in zip(junction_coords[0], junction_coords[1]):
        cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

    # Draw endpoints in blue
    endpoint_coords = np.where(endpoints)
    for y, x in zip(endpoint_coords[0], endpoint_coords[1]):
        cv2.circle(overlay, (x, y), 3, (255, 0, 0), -1)

    # Combine images
    result = cv2.addWeighted(output, 0.6, overlay, 0.8, 0)

    return result, overlay


def analyze_cell_junctions(image_path, visualize=True):
    """Main function to analyze cell junctions with improved cleaning"""
    print("Loading and preprocessing image...")
    img, gray = load_and_preprocess(image_path)

    print("Enhancing cell boundaries...")
    enhanced = enhance_junctions(gray)

    print("Creating clean skeleton...")
    skeleton = create_clean_skeleton(enhanced)

    print("Removing border artifacts...")
    skeleton = remove_border_endpoints(skeleton, border_margin=10)

    print("Classifying skeleton pixels...")
    junction_points, edge_points, endpoints, neighbor_count = classify_skeleton_pixels(skeleton)

    print("Removing junction-endpoint branches...")
    skeleton_final = remove_junction_endpoint_branches(skeleton)

    # Re-classify after final cleaning
    junction_points, edge_points, endpoints, neighbor_count = classify_skeleton_pixels(skeleton_final)

    print("Counting edge segments...")
    num_edge_segments, labeled_edges = count_edge_segments(skeleton_final, junction_points)

    # Count actual junction points (cluster junction pixels)
    num_tricellular, junction_representatives = count_actual_junction_points(junction_points)

    num_endpoints = np.sum(endpoints)
    num_edge_pixels = np.sum(edge_points)

    # Print results
    print(f"Tricellular junctions (3+ cells meet): {num_tricellular}")
    print(f"Edge segments (bicellular junctions): {num_edge_segments}")
    print(f"Endpoints: {num_endpoints}")
    print(f"Edge pixels: {num_edge_pixels}")
    print(f"Total skeleton pixels: {np.sum(skeleton_final)}")

    if visualize:
        # Create visualization
        result_img, overlay = visualize_results(img, skeleton_final, junction_representatives, edge_points, endpoints)

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Enhanced image
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced Boundaries')
        axes[0, 1].axis('off')

        # Clean skeleton
        axes[0, 2].imshow(skeleton_final, cmap='gray')
        axes[0, 2].set_title('Clean Skeleton')
        axes[0, 2].axis('off')

        # Neighbor count heatmap
        axes[1, 0].imshow(neighbor_count, cmap='hot')
        axes[1, 0].set_title('Neighbor Count Heatmap')
        axes[1, 0].axis('off')

        # Junction classification - skeleton with colored points
        skeleton_with_points = np.zeros((skeleton_final.shape[0], skeleton_final.shape[1], 3), dtype=np.uint8)
        # White skeleton
        skeleton_with_points[skeleton_final] = [255, 255, 255]

        # Mark junction points in red
        junction_coords = np.where(junction_points)
        for y, x in zip(junction_coords[0], junction_coords[1]):
            cv2.circle(skeleton_with_points, (x, y), 4, (255, 0, 0), -1)  # Red

        # Mark endpoints in blue
        endpoint_coords = np.where(endpoints)
        for y, x in zip(endpoint_coords[0], endpoint_coords[1]):
            cv2.circle(skeleton_with_points, (x, y), 3, (0, 0, 255), -1)  # Blue

        axes[1, 1].imshow(skeleton_with_points)
        axes[1, 1].set_title('Skeleton with Junction Points\n(Red: Junctions, Blue: Endpoints)')
        axes[1, 1].axis('off')

        # Final result
        axes[1, 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Final Result\nJunctions: {num_tricellular}, Edges: {num_edge_segments}')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save result
        cv2.imwrite('clean_junction_detection.png', result_img)
        print("\nResult saved as 'clean_junction_detection.png'")

    return {
        'tricellular_junctions': num_tricellular,
        'edge_segments': num_edge_segments,
        'endpoints': num_endpoints,
        'skeleton': skeleton_final,
        'junction_points': junction_points,
        'edge_points': edge_points
    }


if __name__ == "__main__":
    # Replace with your image path
    image_path = "im_test.jpg"

    try:
        results = analyze_cell_junctions(image_path, visualize=True)
        print(f"\nFinal counts:")
        print(f"Tricellular junctions: {results['tricellular_junctions']}")
        print(f"Bicellular edge segments: {results['edge_segments']}")

    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        print("Please update the image_path variable with the correct path to your image.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure you have all required libraries installed:")
        print("pip install opencv-python numpy scikit-image matplotlib scipy")