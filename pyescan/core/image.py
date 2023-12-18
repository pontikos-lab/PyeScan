from cached_property import cached_property

from .utils import ArrayView

import numpy as np
from PIL import Image as PILImage

class LazyImage():
    """
    Holder for single image to enable lazy loading
    """
    def __init__(self, file_path=None, mode=None):
        self._source_type = None # For future use to manage different types of loading
        self._file_location = file_path
        
        self._color_mode = mode
        
        # Set initial params
        self._raw_image = None
        self._image = None
        self.loaded = False # Currently checking for _image is None

    def __getattr__(self, attr):
        # Pass through other calls to underlying image
        return getattr(self.image, attr)
    
    def __array__(self):
        return self.data
    
    def _repr_png_(self):
        return self.image._repr_png_()
    
    def load(self, force=False):
        #TODO: add proper debug logging
        #print("Loaded image", self._file_location)
        self._raw_image = PILImage.open(self._file_location)
        self._image = self._raw_image
        if self._color_mode:
            self._image = self._image.convert(mode=self._color_mode)
        self.loaded = True
        
    def reload(self):
        # Forces reload
        self.load(force=True)
        
    def unload(self):
        if self._image is not None:
            self._image.close()
            self._image = None
            self._raw_image = None
        self.loaded = False
        
    @property
    def image(self):
        if self._image is None:
            self.load()
        return self._image
    
    @property
    def data(self):
        return np.array(self.image)
    


class ImageVolume(ArrayView):
    """
    Holder for set of images
    """
    def __init__(self, images=None, file_paths=None, mode=None):
        if not images is None:
            self._images = images
        else:
            self._images = [ LazyImage(file_path, mode) for file_path in file_paths ]
        
    def _items(self):
        return self._images
