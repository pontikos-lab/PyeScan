from IPython.display import display
from numpy.typing import NDArray

from .scan import SingleImageScan
from .visualisation import enface_display_widget, overlay_masks

        
class EnfaceScan(SingleImageScan):
    """
    Base class for scan as a single image / bscan
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _annotated_enface(self, features=None) -> NDArray:
        image = self.image
        masks = [annotation.images.get(0, None) for annotation in self.annotations.values()]
        annotated_image = overlay_masks(image, masks, feature_names=self.annotations.keys(), alpha=0.5)
        return annotated_image # Should maybe convert to PIL image
    
    def _build_display_widget(self):
        if self.annotations:
            enface_image = self._annotated_enface()
        else:
            enface_image = self.image
        return enface_display_widget(enface_image, width=320, height=320)
    
    def _ipython_display_(self):
        display(self._build_display_widget())
        
class FAFScan(EnfaceScan):
    """
    Class for FAF
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class IRScan(EnfaceScan):
    """
    Class for IR
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class ColorFundusScan(EnfaceScan):
    """
    Class for Color Fundus (maybe can have a separate for OPTOS)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

