from abc import ABCMeta, abstractmethod, abstractproperty
from cached_property import cached_property

from .image import LazyImage, ImageVolume
from .utils import ArrayView

class ModelInfo():
    """
    Datastructure for holding information about a model
    """
    def __init__(self):
        self.name = None
    
class FeatureInfo():
    """
    Datastructure for holding information about a model 
    """
    def __init__(self):
        self.name = None
        
class MaskImage(LazyImage):
    @property
    def image(self):
        # Add exception for Missing filepath as we don't want to throw exception
        if self._file_location is None:
            return None
        return super().image

# Could probably just use a transparent arrayview here, rather than parent class
# TODO: Handle missing masks gracefully
class MaskVolume(ArrayView):
    """
    Maybe slightly pointless wrapper for array of bscans
    """
    def __init__(self, masks):
        self._masks = masks
        
    def _items(self):
        return self._masks
            
    def _repr_png_(self):
        return self._masks[len(self._masks)//2]._repr_png_()
    
    def preload(self):
        for mask in self._masks:
            mask.preload()
        
    def unload(self):
        for mask in self._masks:
            mask.unload()

    @property
    def images(self):
        return ImageVolume([mask.image for mask in self._masks])
    

class Annotation(metaclass=ABCMeta):
    """
    Base class for holding annotation information
    """
    def __init__(self, basescan=None, feature=None, modelinfo=None):
        self._scan = basescan
        self._feature = feature
        self._modelinfo = modelinfo
        
        self._mask = None
        self._threshold = 0.5
        
    #@abstractmethod
    def get_area_px(self):
        raise NotImplementedError()
        
    #@abstractmethod
    def get_area_mm(self):
        raise NotImplementedError()
        
    #@abstractmethod
    def get_volume_px(self):
        raise NotImplementedError()
        
    #@abstractmethod
    def get_volume_mm(self):
        raise NotImplementedError()
        
    @abstractproperty
    def data(self):
        raise NotImplementedError()
    
    
class AnnotationEnface(Annotation):
    """
    Base class for holding annotation information
    """
    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask = mask #MaskImage
        
        
class AnnotationOCT(Annotation):
    """
    Base class for holding annotation information
    """
    def __init__(self, masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._masks = masks #MaskVolume
        
    def _repr_png_(self):
        return self._masks._repr_png_()
    
    def _ipython_display_(self):
        #print("Test")
        from IPython.display import display, Image
        display(self._build_display_widget())
        
    @property
    def images(self):
        return self._masks.images
    
    @property
    def data(self):
        return self._masks.data
   
    def _build_display_widget(self):
        from ipywidgets import widgets
        from PIL import Image as PILImage
        import io
        
        width=320
        height=320
        
        def encode_image(image):
            # Save image to buffer
            imgByteArr = io.BytesIO()
            if image is None:
                image = PILImage.new('L', (20, 20))
            image.save(imgByteArr, format='PNG')
            # Turn the BytesIO object back into a bytes object
            imgByteArr = imgByteArr.getvalue()
            return imgByteArr
        
        encoded_volume = [ encode_image(image) for image in self.images ]
        n_images = len(encoded_volume)

        # Create a slider widget for image navigation
        w_slider = widgets.IntSlider(min=0, max=n_images-1, step=1,
                                         layout={'width': str(width*2)+'px'},
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
        
        # Display the widgets
        #img_display = display(image_layout, display_id=True)
        return display_layout
        