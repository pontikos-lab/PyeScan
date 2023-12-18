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
    
    def _ipython_display_(self):
        #print("Test")
        from IPython.display import display, Image
        #display(Image(self._repr_png_()))
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
        return self._scans
        
        
    def _build_display_widget(self):
        from ipywidgets import widgets
        from PIL import Image as PILImage
        import io
        def encode_image(image):
            #TODO: Enable this conversion to make this more general in future
            #image = PILImage(image)

            # Save image to buffer
            imgByteArr = io.BytesIO()
            image.save(imgByteArr, format=image.format)
            # Turn the BytesIO object back into a bytes object
            imgByteArr = imgByteArr.getvalue()
            return imgByteArr
        
        width=320
        height=320
        
        encoded_enface = encode_image(self.enface.image)
        encoded_volume = [ encode_image(image) for image in self.images ]
        n_images = len(encoded_volume)

        # Create a slider widget for image navigation
        w_slider = widgets.IntSlider(min=0, max=n_images-1, step=1,
                                         layout={'width': str(width*2)+'px'},
                                         readout=True,
                                         readout_format='d')

        w_value = widgets.Label(value="Some text",
                                layout={'width': str(width*2)+'px', 'visible': 'true'})
        
        # Create an image widget for displaying images
        w_image_enface = widgets.Image(value=encoded_enface, width=width, height=height)

        # Create an image widget for displaying images
        w_image_volume = widgets.Image(value=encoded_volume[n_images//2], width=width, height=height)
        
        # Define a function to update the displayed image based on the slider value
        def update_image(change):
            index = change.new
            w_image_volume.value=encoded_volume[index]
            #image_widget.reload() #Not currenly needed
            #img_display.update(image_layout) #Not currently needed

        # Connect the slider and image widgets
        w_slider.observe(update_image, names='value')

        # Arrange the widgets using an HBox
        image_layout = widgets.HBox([w_image_enface, w_image_volume])
        display_layout = widgets.VBox([w_slider, image_layout])
        
        # Display the widgets
        #img_display = display(image_layout, display_id=True)
        return display_layout