def get_mask_stats(row):
    from PIL import Image
    import numpy as np
    try:
        img = Image.open(row.file_path).convert("L")
        width, height = img.size
        mask = np.array(img) > 127
        return width, height, mask.sum(), mask.any(axis=0).sum(), mask.any(axis=1).sum()
    except FileNotFoundError as e: 
        return None, None, 0., 0., 0.
    
def get_mask_area(row):
    rescale_w = row.size_width / row.mask_width
    rescale_h = row.size_height / row.mask_height
    
    area = row.pixel_count * row.resolutions_mm_width * row.resolutions_mm_height * rescale_w * rescale_h
    horizontal_extent = row.horizontal_pixel_count * row.resolutions_mm_width * rescale_w
    vertical_extent = row.vertical_pixel_count * row.resolutions_mm_height * rescale_h
    return area, horizontal_extent, vertical_extent

def count_mask_clusters(row):
    import cv2
    import numpy as np

    # Read the binary mask
    binary_mask = cv2.imread(row.file_path, 0)

    # Preprocessing steps to remove noise and enhance the segmentation
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the watershed
    binary_mask = cv2.imread(row.file_path)
    cv2.watershed(binary_mask, markers)

    # Find the number of clusters
    unique_markers = np.unique(markers)
    # Removing -1 and 1 as they represent boundaries and background
    unique_markers = [marker for marker in unique_markers if marker not in [-1, 1]]

    # Print the number of clusters
    return len(unique_markers)

def get_circle_line_intersection(circle_centre, radius, line_start, line_end):
    import math

    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    
    offset_x = line_start[0] - circle_centre[0]
    offset_y = line_start[1] - circle_centre[1]
    
    # Formula from ChatGPT
    # Could look up derivation, should probably double check
    # Is presumably line given as interpolation with t, then solve for t
    # in circle eqn
    a = dx**2 + dy**2
    b = 2 * (dx * offset_x + dy * offset_y)
    c = offset_x**2 + offset_y**2 - radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return []

    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)

    return t1, t2

def get_distance_mask(row, radius):
    bscan_start = 0, row.bscan_index * row.resolutions_mm_depth
    bscan_end = row.dimensions_mm_width, row.bscan_index * row.resolutions_mm_depth

    fovea_location = row.fovea_x * row.resolutions_mm_width, \
                     row.fovea_bscan_index * row.resolutions_mm_depth
    
    intersection = get_circle_line_intersection(fovea_location, radius, bscan_start, bscan_end)

    im_w = row.size_width
    im_h = row.size_height
    mask = np.zeros((im_h, im_w))
    
    if intersection:
        start = max(0.0, intersection[0])
        end = min(1.0, intersection[1])
        mask[:,int(start*im_w):int(end*im_w)-1] = 1.0
    return mask

def get_volume_at_distance(row, radius, area_only=False):
    mask = get_distance_mask(row, radius)
    if not mask.any(): return 0
    if mask.all(): return row.pixel_count_horizontal if area_only else row.pixel_count
    
    img = np.array(Image.open(row.file_path)) > 0.
    masked_img = img * mask
    
    if area_only:
        return masked_img.any(axis=0).sum()
    return masked_img.sum()

def get_1mm_ring_volume(row):
    return get_volume_at_distance(row, .5, area_only=False)

def get_3mm_ring_volume(row):
    return get_volume_at_distance(row, 1.5, area_only=False)

def get_6mm_ring_volume(row):
    return get_volume_at_distance(row, 3., area_only=False)

def get_1mm_ring_area(row):
    return get_volume_at_distance(row, .5, area_only=True)

def get_3mm_ring_area(row):
    return get_volume_at_distance(row, 1.5, area_only=True)

def get_6mm_ring_area(row):
    return get_volume_at_distance(row, 3., area_only=True)