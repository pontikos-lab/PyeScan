import numpy as np
import PIL.Image
from typing import List, Dict, Tuple, Any, Optional
from .registry import pyescan_metric, Meta, Stat, Spec, Pred

@pyescan_metric()
def get_mask(file_path_mask: str) -> Tuple[Spec[np.array]]:
    try:
        mask = np.array(PIL.Image.open(file_path_mask).convert("L")) > 0
        return mask,
    except FileNotFoundError as e: 
        return None,
    
@pyescan_metric()
def get_image(file_path_scan: str) -> Tuple[Spec[np.array]]:
    try:
        image = np.array(PIL.Image.open(file_path_scan).convert("L"))
        return image,
    except FileNotFoundError as e: 
        return None,

@pyescan_metric()
def get_mask_shape(mask: Spec[np.array]) -> Tuple[Stat[int], Stat[int], Stat[float], Stat[float], Stat[float]]:
    mask_height_px, mask_width_px = mask.shape
    return mask_width_px, mask_height_px

@pyescan_metric()
def get_mask_pixel_counts(mask: Spec[np.array]) -> Tuple[Stat[float], Stat[float], Stat[float]]:
    pixel_count, columns_count, rows_count = mask.sum(), mask.any(axis=0).sum(), mask.any(axis=1).sum()
    return pixel_count, columns_count, rows_count

@pyescan_metric()
def get_mask_area(
        scan_width_px: Meta[int],
        scan_height_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_height: Meta[float],
        mask_width_px: Stat[int],
        mask_height_px: Stat[int],
        pixel_count: Stat[int],
        columns_count: Stat[int],
        rows_count: Stat[int]
    ) -> Tuple[Stat[float], Stat[float], Stat[float]]:

    if mask_width_px is None: return None, None, None
    
    rescale_w, rescale_h = scan_width_px / mask_width_px, scan_height_px / mask_height_px
    mask_area = pixel_count * resolutions_mm_width * resolutions_mm_height * rescale_w * rescale_h
    horizontal_extent = columns_count * resolutions_mm_width * rescale_w
    vertical_extent = rows_count * resolutions_mm_height * rescale_h
    return mask_area, horizontal_extent, vertical_extent


@pyescan_metric()
def infer_fovea_enface_position(
        bscan_index: Meta[int],
        scan_width_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_depth: Meta[float],
        bscan_location_start_x: Meta[int],
        bscan_location_start_y: Meta[int],
        bscan_location_end_x: Meta[int],
        bscan_location_end_y: Meta[int],
        mask_width_px: Stat[int],
        mask_height_px: Stat[int],
        fovea_x: Pred[int],
        fovea_bscan_index: Pred[int],
    ):
    # Calculates enface x,y position of fovea using information from current bscan only
    import numpy as np
    
    # Get bscan start and end positions of current b-scan
    bscan_start_x, bscan_start_y = bscan_location_start_x, bscan_location_start_y
    bscan_end_x, bscan_end_y = bscan_location_end_x, bscan_location_end_y
    bscan_width = scan_width_px * resolutions_mm_width # prefer using 
    bscan_depth = resolutions_mm_depth
    
    # Get fovea's OCT coordinates
    fovea_u, fovea_v = fovea_x, fovea_bscan_index
    
    # Calculate fovea enface x and y by projecting from current b-scan
    bscan_dir_x = bscan_end_x - bscan_start_x
    bscan_dir_y = bscan_end_y - bscan_start_y
    fovea_t = fovea_u / scan_width_px
    fovea_projection_x = bscan_start_x + fovea_t * bscan_dir_x
    fovea_projection_y = bscan_start_y + fovea_t * bscan_dir_y
    
    # Use the enface length to estimate the bscan spacing in enface px
    bscan_length = np.sqrt(bscan_dir_x**2 + bscan_dir_y**2)
    enface_pixel_pex_mm = bscan_length / bscan_width
    bscan_spacing = enface_pixel_pex_mm * bscan_depth
    perp_length = ( bscan_index - fovea_v ) * bscan_spacing
    
    # Add perpendicular component
    fovea_enface_x = fovea_projection_x - perp_length * (bscan_dir_y / bscan_length)
    fovea_enface_y = fovea_projection_y + perp_length * (bscan_dir_x / bscan_length)
    return fovea_enface_x, fovea_enface_y


@pyescan_metric(
  returns = ["distance_mask_enface_<diameter>mm"],
    parameters = ["diameter"],
)
def _get_distance_mask_enface(
        scan_width_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_height: Meta[float],
        mask_width_px: Stat[int],
        mask_height_px: Stat[int],
        fovea_enface_x: Pred[int],
        fovea_enface_y: Pred[int],
        diameter: float,
    ) -> Tuple[Spec[np.array]]:
    from skimage.draw import disk

    radius  = float(diameter) / 2
    radius_scale_factor = mask_width_px / scan_width_px
    
    radius_scaled = radius / resolutions_mm_width #assuming the same in both dims
    radius_scaled *= radius_scale_factor
    
    mask = np.zeros((mask_height_px, mask_width_px))
    center = fovea_enface_y, fovea_enface_x # y, x (y=0 top)
    shape = mask_height_px, mask_width_px # h, w
    rr, cc = disk(center, radius_scaled, shape=shape)
    mask[rr,cc] = 1.
    
    distance_mask_enface = mask
    
    return distance_mask_enface,


@pyescan_metric()
def _get_quadrant_masks_enface(
        mask_width_px: Stat[int],
        mask_height_px: Stat[int],
        fovea_enface_x: Pred[int],
        fovea_enface_y: Pred[int]
    ) -> Tuple[Spec[np.array], Spec[np.array], Spec[np.array], Spec[np.array]]:
    import numpy as np
    # Should probably normalise fovea position by actual image size...

    im_w = mask_width_px
    im_h = mask_height_px
    
    center_x, center_y = fovea_enface_x, fovea_enface_y
    
    y, x = np.ogrid[:im_h, :im_w]
    upper_left = (y - center_y >= x - center_x)
    upper_right = (y - center_y >= center_x - x)
    
    masks = [ np.zeros((im_h, im_w)) for _ in range(4) ]
    masks[0][~upper_left & ~upper_right] = 1.0 # Upper, increasing y is bottom
    masks[1][~upper_left & upper_right] = 1.0 # Right (as viewed)
    masks[2][upper_left & upper_right] = 1.0 # Lower
    masks[3][upper_left & ~upper_right] = 1.0 # left (as viewed)
    
    quadrant_mask_enface_superior, quadrant_mask_enface_dexter, \
    quadrant_mask_enface_inferior, quadrant_mask_enface_sinister = masks
    return ( quadrant_mask_enface_superior,
             quadrant_mask_enface_dexter,
             quadrant_mask_enface_inferior,
             quadrant_mask_enface_sinister )

@pyescan_metric(
    requires=["spec:mask", "stat:quadrant_mask_enface_<quadrant>"],
    returns=["pixel_count_enface_<quadrant>"],
    parameters = ["quadrant"],
)
def get_pixel_count_by_quadrant_enface(
        mask: Spec[np.array],
        quadrant_mask_enface: Spec[np.array],
        quadrant: str,
    ) -> Tuple[Stat[int], Stat[int], Stat[int], Stat[int]]:
    masked_img = mask * quadrant_mask_enface
    result = masked_img.sum()
    return pixel_count,

@pyescan_metric(
    requires=["spec:mask", "stat:distance_mask_enface_<diameter>mm"],
    returns=["pixel_count_enface_<diameter>mm"],
    parameters = ["diameter"],
)
def get_pixel_count_by_distance_enface(
        mask,
        distance_mask_enface: Spec[np.array],
        diameter: float,
    ) -> Tuple[Stat[int]]:
    masked_img = mask * distance_mask_enface
    pixel_count = masked_img.sum()
    return pixel_count,

@pyescan_metric(
    requires=[
        "spec:mask",
        "spec:distance_mask_enface_<diameter>mm",
        "spec:quadrant_mask_enface_<quadrant>"
    ],
    returns=["pixel_count_enface_<diameter>mm_<quadrant>"],
    parameters = ["diameter", "quadrant"],
)
def get_pixel_count_by_distance_quadrant_enface(
        mask,
        distance_mask_enface: Spec[np.array],
        quadrant_mask_enface: Spec[np.array],
        diameter: float,
        quadrant: str,
    ) -> Tuple[Stat[int]]:
    masked_img = mask * distance_mask_enface * quadrant_mask_enface
    pixel_count = masked_img.sum()
    return pixel_count,


def _get_circle_line_intersection(circle_centre, radius, line_start, line_end):
    # Used for distance mask calculations
    # TODO: Move to utils?
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

@pyescan_metric(
    returns = ["distance_mask_oct_<diameter>mm"],
    parameters = ["diameter"],
)
def _get_distance_mask_oct(
        bscan_index: Meta[int],
        scan_width_px: Meta[int],
        resolutions_mm_width: Meta[float],
        resolutions_mm_depth: Meta[float],
        mask_width_px: Stat[int],
        mask_height_px: Stat[int],
        fovea_x: Pred[int],
        fovea_bscan_index: Pred[int],
        diameter: float,
    ) -> Tuple[Spec[np.array]]:
    
    import numpy as np
    radius = float(diameter) / 2.
    
    bscan_start = 0, bscan_index * resolutions_mm_depth
    bscan_end = scan_width_px * resolutions_mm_width, bscan_index * resolutions_mm_depth

    fovea_location = fovea_x * resolutions_mm_width, \
                     fovea_bscan_index * resolutions_mm_depth
    
    intersection = _get_circle_line_intersection(fovea_location, radius, bscan_start, bscan_end)

    mask = np.zeros((mask_height_px, mask_width_px))
    if intersection:
        start = max(0.0, intersection[0])
        end = min(1.0, intersection[1])
        mask[:,int(start*mask_width_px):int(end*mask_width_px)-1] = 1.0
    distance_mask_oct = mask
    return distance_mask_oct,

@pyescan_metric()
def _get_quadrant_masks_oct(
        bscan_location_start_x: Meta[int],
        bscan_location_start_y: Meta[int],
        bscan_location_end_x: Meta[int],
        bscan_location_end_y: Meta[int],
        mask_width_px: Stat[int],
        mask_height_px: Stat[int],
        fovea_enface_x: Pred[int],
        fovea_enface_y: Pred[int],
    ) -> Tuple[Spec[np.array], Spec[np.array], Spec[np.array], Spec[np.array]]:
    import numpy as np

    # Get bscan start and end positions of current b-scan
    bscan_start_x, bscan_start_y = bscan_location_start_x, bscan_location_start_y
    bscan_end_x, bscan_end_y = bscan_location_end_x, bscan_location_end_y
    
    # Get B-scan's start/end points relative to fovea
    start_x, start_y = bscan_start_x - fovea_enface_x, bscan_start_y - fovea_enface_y
    end_x, end_y = bscan_end_x - fovea_enface_x, bscan_end_y - fovea_enface_y

    # Do co-ordinate shift w=y-x, z=y+x (equiv to rotation + scale)
    start_w, start_z = start_y - start_x, start_x + start_y
    end_w, end_z = end_y - end_x, end_x + end_y
    
    # upper left is x < y, upper right is y > -x or -y < x
    # rearranging for co-ord transfm 0 < (y-x) and 0 < (x+y)

    # Calculate mask of upper and left quadrants, y > x, w > 0
    upper_left = np.zeros((mask_height_px, mask_width_px))
    v_intercept = - start_z / (end_z - start_z) # w-axis intercept
    if 0 <= v_intercept <= 1:
        x_intercept = int(v_intercept * (mask_width_px - 1))
        if end_z > start_z:
            upper_left[...,x_intercept:] = 1.
        else:
            upper_left[...,:x_intercept] = 1.
    elif start_z > 0:
        upper_left[...] = 1.
        
    # Calculate mask of upper and right quadrants, y > -x, z > 0
    upper_right = np.zeros((mask_height_px, mask_width_px))
    v_intercept = - start_w / (end_w - start_w) # z-axis intercept
    if 0 <= v_intercept <= 1:
        x_intercept = int(v_intercept * (mask_width_px - 1))
        if end_w > start_w:
            upper_right[...,x_intercept:] = 1.
        else:
            upper_right[...,:x_intercept] = 1.
    elif start_w > 0:
        upper_right[...] = 1.
    
    # Idea - make masks for upper left and upper right and multiply them
    masks = np.zeros((4, mask_height_px, mask_width_px))
    masks[0,...] = upper_left * upper_right # Upper
    masks[1,...] = upper_left * (1-upper_right) # Right (as viewed)
    masks[2,...] = (1-upper_left) * (1-upper_right) # Lower
    masks[3,...] = (1-upper_left) * upper_right # left (as viewed)
    quadrant_mask_oct_superior, quadrant_mask_oct_dexter, \
    quadrant_mask_oct_inferior, quadrant_mask_oct_sinister = masks
    return ( quadrant_mask_oct_superior,
             quadrant_mask_oct_dexter,
             quadrant_mask_oct_inferior,
             quadrant_mask_oct_sinister )

@pyescan_metric(
    requires=["spec:mask", "stat:quadrant_mask_oct_<quadrant>"],
    returns=["pixel_count_oct_<quadrant>"],
    parameters = ["quadrant"],
)
def get_pixel_count_by_quadrant_oct(
        mask: Spec[np.array],
        quadrant_mask_oct: Spec[np.array],
        quadrant: str,
    ) -> Tuple[Stat[int]]:
    masked_img = mask * quadrant_mask_oct
    pixel_count = masked_img.sum()
    return pixel_count,

@pyescan_metric(
    requires=["spec:mask", "stat:distance_mask_oct_<diameter>mm"],
    returns=["pixel_count_oct_<diameter>mm"],
    parameters = ["diameter"],
)
def get_pixel_count_by_distance_oct(
        mask,
        distance_mask_oct: Spec[np.array],
        diameter: float,
    ) -> Tuple[Stat[int]]:
    masked_img = mask * distance_mask_oct
    pixel_count = masked_img.sum()
        
    return pixel_count,

@pyescan_metric(
    requires=[
        "spec:mask",
        "spec:distance_mask_oct_<diameter>mm",
        "spec:quadrant_mask_oct_<quadrant>"
    ],
    returns=[
        "pixel_count_oct_<diameter>mm_<quadrant>",
    ],
    parameters = ["diameter", "quadrant"],
)
def get_pixel_count_by_distance_quadrant_oct(
        mask,
        distance_mask_oct: Spec[np.array],
        quadrant_mask_oct: Spec[np.array],
        diameter: float,
        quadrant: str,
    ) -> Tuple[Stat[int],]:
    masked_img = mask * distance_mask_oct * quadrant_mask_oct
    pixel_count = masked_img.sum()
    return pixel_count,

