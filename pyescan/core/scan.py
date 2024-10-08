from abc import ABCMeta, abstractmethod, abstractproperty
from cached_property import cached_property

class BaseScan(metaclass=ABCMeta): # We could probably make this into a metaclass itself
    def __init__(self, metadata=None, *args, **kwargs):
        self._record = None
        self._group_id = None # This should uniquely identify the high-level scan within the record
        
        self._metadata = metadata 

        self._annotations = dict() # Keep a reference to annotations in the class?
    
    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return str(self.__repr__())
    
    @abstractmethod
    def plot_image(self, include_annotations=False):
        raise NotImplementedError()
    
    @abstractmethod
    def preload(self):
        """
        Force load of all images/masks/etc, may be useful for dataloading
        """
        raise NotImplementedError()
    
    @abstractmethod
    def unload(self):
        """
        Force unload of all images/masks/etc, could be useful for saving memory
        """
        raise NotImplementedError()
        
    #@abstractmethod
    def _save(self, path_to_output_folder, format=None):
        raise NotImplementedError()
        
    def find_annotations(self):
        """ Try search for annotations automatically """
        raise NotImplementedError()
        
    def add_annotation(self, feature_name, annotation):
        annotation._scan = self
        self._annotations[feature_name] = annotation
        
    def add_annotations(self, annotation_dict):
        for feature_name, annotation in annotation_dict.items():
            annotation._scan = self
            self._annotations[feature_name] = annotation
        
    @property
    def annotations(self):
        return self._annotations

    @abstractproperty
    def image(self):
        raise NotImplementedError()

    @abstractproperty
    def data(self):
        raise NotImplementedError()
        
    @property
    def metadata(self):
        return self._metadata

    @cached_property
    def modality(self):
        # Can maybe return the class name?
        return self._metadata.modality

    @cached_property
    def patient_id(self):
        raise NotImplementedError()
        
    @cached_property
    def scan_id(self):
        raise NotImplementedError()
        
    @cached_property
    def group_id(self):
        return self.metadata.group
        
    @cached_property
    def source_id(self):
        return self.metadata.source_id
        
    @cached_property
    def laterality(self):
        return self.metadata.laterality

    @cached_property
    def scan_angle(self):
        raise NotImplementedError()
        
    
class SingleImageScan(BaseScan):
    """
    Base class for scan as a single image / bscan
    """
    def __init__(self, image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image = image
        
    def _repr_png_(self):
        return self._image._repr_png_()
    
    def __array__(self):
        return self._image.data
      
    @property
    def image(self):
        return self._image
    
    @property
    def data(self):
        return self._image.data
    
    @property
    def shape(self):
        return self._image.data.shape
    
    def plot_image(self, include_annotations=False):
        raise NotImplementedError()
    
    def preload(self):
        self._image.load()
    
    def unload(self):
        self._image.unload()

    
    