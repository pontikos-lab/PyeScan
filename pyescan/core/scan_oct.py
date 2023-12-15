from .image import ImageVolume
from .scan import BaseScan, SingleImageScan
from .utils import ArrayView

class BScan(SingleImageScan):
    """
    Class for single OCT b-scan
    """
    def __init__(self, image, bscan_index, *args, **kwargs):
        super().__init__(image, *args, **kwargs)
        self._scan_index = bscan_index

class BScanArray(ArrayView):
    """
    Maybe slightly pointless wrapper for array of bscans
    """
    def __init__(self, bscans):
        self._bscans = bscans
        
    def _items(self):
        return self._bscans
            
    def _repr_png_(self):
        return self._bscans[len(self._bscans)//2]._repr_png_()
    
    def __getitem__(self, index):
        return self._bscans[index]
    
    def __len__(self):
        return len(self._bscans)
    
    def __array__(self):
        return self._bscans.data
    
    def preload(self):
        for bscan in self._bscans:
            bscan.preload()
        
    def unload(self):
        for bscan in self._bscans:
            bscan.unload()

    @property
    def images(self):
        return ImageVolume([bscan._image for bscan in self._bscans])
    
class OCTScan(BaseScan):
    """
    Class for OCT scans
    """
    def __init__(self, enface, bscans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enface = enface
        self._bscans = bscans
        
    def _repr_png_(self):
        return self._bscans._repr_png_()
    
    def __getitem__(self, index):
        return self._bscans[index]
    
    def __len__(self):
        return len(self._bscans)
    
    def __array__(self):
        return self._bscans.data
    
        
    def preload(self):
        self._enface.preload()
        self._scans.preload()
    
    def unload(self):
        self._enface.unload()
        self._scans.unload()
        
    @property
    def image(self):
        return self._enface.image
    
    @property
    def images(self):
        return self._bscans.images
    
    @property
    def data(self):
        return self._bscans.data
    
    @property
    def shape(self):
        return self._bscans.data.shape
    
    def plot_image(self, include_annotations=False):
        raise NotImplementedError()

    @property
    def enface(self):
        return self._enface

    @property
    def bscans(self):
        return self._scans
        
    