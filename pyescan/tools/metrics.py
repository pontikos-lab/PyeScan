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
    #Using watershet clustering
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
    if row.modality == "OCT":
        return get_distance_mask_oct(row, radius)
    else:
        return get_distance_mask_enface(row, radius)

def get_distance_mask_oct(row, radius):
    import numpy as np
    bscan_start = 0, row.bscan_index * row.resolutions_mm_depth
    bscan_end = row.dimensions_mm_width, row.bscan_index * row.resolutions_mm_depth

    fovea_location = row.fovea_x * row.resolutions_mm_width, \
                     row.fovea_bscan_index * row.resolutions_mm_depth
    
    intersection = get_circle_line_intersection(fovea_location, radius, bscan_start, bscan_end)

    im_w = row.mask_width
    im_h = row.mask_height
    mask = np.zeros((im_h, im_w))
    
    if intersection:
        start = max(0.0, intersection[0])
        end = min(1.0, intersection[1])
        mask[:,int(start*im_w):int(end*im_w)-1] = 1.0
    return mask

def get_distance_mask_enface(row, radius):
    import numpy as np
    from skimage.draw import disk

    im_w = row.mask_width
    im_h = row.mask_height
    mask = np.zeros((im_h, im_w))
    
    radius_scale_factor = im_w / row.size_width
    
    if False:#radius_scale_factor != im_h / row.size_height: # different aspect ratio not supported
        print(row.file_path)
        print("Mask:", row.mask_width, row.mask_height)
        print("Actual:", row.size_width, row.size_height)
        #raise Exception()
    
    if not row.resolutions_mm_width: return 0
    radius_scaled = radius / row.resolutions_mm_width #assuming the same in both dims
    radius_scaled *= radius_scale_factor
    
    if 'fovea_x' in row:
        fovea_location = row.fovea_x, row.fovea_y
    else:
        fovea_location = im_w // 2, im_h //2
        
    
    center = fovea_location[1], fovea_location[0] # y, x (y=0 top)
    shape = im_h, im_w # h, w
    rr, cc = disk(center, radius_scaled, shape=shape)
    mask[rr,cc] = 1.
    
    return mask

def get_pixel_count_at_distance(row, radius, hexent_only=False):
    import numpy as np
    from PIL import Image
    
    import numbers
    return_single_value = False
    if isinstance(radius, numbers.Number):
        radius = [ radius ]
        return_single_value = True
    
    masks = [ get_distance_mask(row, rad) for rad in radius ]
    mask = np.array(masks)
    if not mask.any():
        return 0. if return_single_value else [ 0. ] * len(radius)
    if mask.all():
        val = row.pixel_count_horizontal if hexent_only else row.pixel_count
        return val if return_single_value else [ val ] * len(radius)

    img = np.array(Image.open(row.file_path).convert('L')) > 0.
    
    results = list()
    for mask in masks:
        masked_img = img * mask
        result = masked_img.any(axis=0).sum() if hexent_only else masked_img.sum()
        results.append(result)
    return results[0] if return_single_value else results

def get_distance_pixel_counts(row, distances=[.5, 1.5, 3.]):
    return get_pixel_count_at_distance(row, distances, hextent_only=False)

def get_distance_volumes(row, distances=[.5, 1.5, 3.]):
    scale_w = row.size_width / row.mask_width * row.resolutions_mm_width
    scale_h = row.size_height / row.mask_height * row.resolutions_mm_height
    scale = scale_w * scale_h * row.resolutions_mm_depth
    
    results = get_pixel_count_at_distance(row, distances, hextent_only=False)
    return [ r*scale for r in results ] 

def get_distance_horizontal_pixel_counts(row, distances=[.5, 1.5, 3.]):
    return get_pixel_count_at_distance(row, distances, hextent_only=True)

def get_distance_areas(row, distances=[.5, 1.5, 3.]):
    hexent_only = (row.modality == "OCT")
    
    if hexent_only:
        scale_w = row.size_width / row.mask_width * row.resolutions_mm_width
        scale = scale_w * row.resolutions_mm_depth
    else:
        scale_w = row.size_width / row.mask_width * row.resolutions_mm_width
        scale_h = row.size_height / row.mask_height * row.resolutions_mm_height
        scale = scale_w * scale_h
    
    results = get_pixel_count_at_distance(row, distances, hexent_only=hexent_only)
    return [ r*scale for r in results ] 