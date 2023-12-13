from abc import ABCMeta, abstractmethod, abstractproperty
from cached_property import cached_property


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

class Annotation(metaclass=ABCMeta):
    """
    Base class for holding annotation information
    """
    def __init__(self):
        self._scan = None
        self._feature = None
        self._modelinfo = None
        
        self._mask = None
        self._threshold = 0.5
        
    @abstractmethod
    def get_area_px(self):
        raise NotImplementedError()
        
    @abstractmethod
    def get_area_mm(self):
        raise NotImplementedError()
        
    @abstractmethod
    def get_volume_px(self):
        raise NotImplementedError()
        
    @abstractmethod
    def get_volume_mm(self):
        raise NotImplementedError()
        
    @abstractmethod
    def data(self):
        raise NotImplementedError()
    

