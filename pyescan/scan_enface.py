from .scan import SingleImageScan
        
class EnfaceScan(SingleImageScan):
    """
    Base class for scan as a single image / bscan
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
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

