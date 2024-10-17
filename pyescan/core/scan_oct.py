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
    
    def preload(self):
        for bscan in self._bscans:
            bscan.preload()
        
    def unload(self):
        for bscan in self._bscans:
            bscan.unload()

    @property
    def images(self):
        return ImageVolume([bscan.image for bscan in self._bscans])
    
    
class OCTScan(BaseScan):
    """
    Class for OCT scans
    """
    def __init__(self, enface, bscans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if enface:
            self._enface = enface #EnfaceScan
            self._enface.set_parent(self)
        
        self._bscans = bscans #BscanArray
        for bscan in self._bscans:
            bscan.set_parent(self)
        
    def _repr_png_(self):
        return self._bscans._repr_png_()
    
    def _ipython_display_(self):
        from IPython.display import display, Image
        display(self._build_display_widget())
    
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
        return self._bscans
    
    def get_bscan_enface_locations(self):
        import numpy as np
        locations = []
        for bscan in self._bscans:
            location_start = (bscan.metadata.bscan_start_x, bscan.metadata.bscan_start_y)
            location_end = (bscan.metadata.bscan_end_x, bscan.metadata.bscan_end_y)
            locations.append((location_start, location_end))
        return np.array(locations)
        
    def _annotatated_bscan(self, bscan_index, features=None):
        pass
        
    def _build_display_widget(self):
        from .visualisation import oct_display_widget, overlay_masks
        
        if self.annotations:
            annotated_images = list()
            for i, image in enumerate(self.images):
                masks = [annotation.images[i] for annotation in self.annotations.values()]
                annotated_image = overlay_masks(image, masks, feature_names=self.annotations.keys(), alpha=0.5)
                annotated_images.append(annotated_image)
        else:
            annotated_images = self.images

        return oct_display_widget(annotated_images, self.enface.image, self.get_bscan_enface_locations(), width=640, height=320, enface_size=320)