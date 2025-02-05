from abc import ABCMeta, abstractmethod, abstractproperty
from cached_property import cached_property
import math

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
        #if self._file_location is None or math.isnan(self._file_location):
        if not isinstance(self._file_location, str):
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
        from IPython.display import display, Image
        display(self._build_display_widget())
        
    @property
    def images(self):
        return self._masks.images
    
    @property
    def data(self):
        return self._masks.data
   
    def _build_display_widget(self):
        from .visualisation import image_array_display_widget
        return image_array_display_widget(self.images, width=320, height=320)
        