def _encode_image(image, default="blank"):
    import io
    from PIL import Image as PILImage
    
    # Save image to buffer
    imgByteArr = io.BytesIO()
    if image is None:
        image = PILImage.new('L', (20, 20))
    image.save(imgByteArr, format='PNG')
    
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

#images = self.images
def image_array_display_widget(images, width=320, height=320):
    from ipywidgets import widgets
    from PIL import Image as PILImage

    encoded_volume = [ _encode_image(image) for image in images ]
    n_images = len(encoded_volume)

    # Create a slider widget for image navigation
    w_slider = widgets.IntSlider(min=0, max=n_images-1, step=1,
                                 layout={'width': str(width)+'px'},
                                 readout=True,
                                 readout_format='d')

    w_value = widgets.Label(value="Some text",
                            layout={'width': str(width)+'px', 'visible': 'true'})

    # Create an image widget for displaying images
    w_image_volume = widgets.Image(value=encoded_volume[n_images//2], width=width, height=height)

    # Define a function to update the displayed image based on the slider value
    def update_image(change):
        index = change.new
        w_image_volume.value=encoded_volume[index]

    # Connect the slider and image widgets
    w_slider.observe(update_image, names='value')

    # Arrange the widgets using a VBox
    display_layout = widgets.VBox([w_slider, w_image_volume])

    return display_layout

#images = self.images
#enface_image = self.enface.image
def oct_display_widget(images, enface_image, width=640, height=320, enface_size=320):
    from ipywidgets import widgets
    from PIL import Image as PILImage

    # Create an image widget for displaying images
    encoded_enface = _encode_image(enface_image)
    w_image_enface = widgets.Image(value=encoded_enface, width=enface_size, height=enface_size)
    
    # Get widget for displaying volume
    w_image_volume = image_array_display_widget(images, width=width, height=height)
    
    # Create layout
    display_layout = widgets.HBox([w_image_enface, w_image_volume])

    return display_layout
    

def generate_distinct_colors(n):
    """
    Generate n distinct colors.
    """
    import colorsys
    
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Slight variation in saturation
        value = 0.8 + (i % 2) * 0.2  # Slight variation in value
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

def overlay_masks(image, masks, colors=None, feature_names=None, alpha=0.5):
    """
    Overlay multiple binary masks on a grayscale image with different transparent colors.
    
    :param image: PIL Image in grayscale mode
    :param masks: List of binary masks (numpy arrays or PIL Images)
    :param colors: Optional list of RGB tuples. If not provided, colors will be auto-generated.
    :param alpha: Transparency of the overlay (0-1), default is 0.5
    :return: PIL Image with overlaid masks
    """
    
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import numpy as np
    
    # Convert grayscale to RGB
    rgb_image = image.convert('RGB')
    
    # Get image dimensions
    width, height = rgb_image.size
    
    # Convert image to numpy array
    img_array = np.array(rgb_image)

    # Generate colors if not provided
    if colors is None:
        colors = generate_distinct_colors(len(masks))

    # Create an empty array for the overlay
    overlay = np.zeros_like(img_array, dtype=np.float32)

    # Process each mask and color
    for mask, color in zip(masks, colors):
        
        # Skip missing masks
        if mask is None: continue
            
        # Ensure mask is a PIL Image
        if isinstance(mask, np.ndarray):
            mask = PILImage.fromarray(mask)
            
        mask = mask.convert('L')
            
        # Resize mask if it doesn't match the input image size
        if mask.size != (width, height):
            mask = mask.resize((width, height), PILImage.NEAREST)
            
        # Convert mask to numpy array and normalize to range [0, 1]
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        # Add colored mask to overlay
        for c in range(3):  # RGB channels
            overlay[:,:,c] += mask_array * color[c] / 255.0

    # Apply alpha blending
    overlay = np.clip(overlay * alpha, 0, 1)

    # Blend original image with overlay
    result = img_array * (1 - overlay) + (overlay * 255)
    result = result.astype(np.uint8)
    
    # Convert back to PIL Image for drawing text
    result_image = PILImage.fromarray(result)
    draw = ImageDraw.Draw(result_image)
    
    if feature_names: 
        # Use default font
        font = ImageFont.load_default()
        
        # Draw labels
        y_offset = 10
        for name, color in zip(feature_names, colors):
            # Draw color swatch
            draw.rectangle([10, y_offset, 30, y_offset + 20], fill=color, outline='white')
            # Draw text
            draw.text((35, y_offset), name, font=font, fill='white', stroke_width=2, stroke_fill='black')
            y_offset += 30

    return result_image