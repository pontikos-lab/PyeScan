"""

HEYEX example format (output of privateEye)

OCT:
- patient
- exam
- series
- images
  - Image 0
    - Source ID
    - modality
    - group
    - size
    - ...
    - contents
    - extras
    - contours
  - Image 1
    - Source ID
    - modality
    - group
    - size
    - ...
    - contents
      - 1
      - 2
      - ...
    - extras
    - contours
- debug
- parser version

Proposed Taxonomy:
0. (Patient)
1. Record - i.e. a single sdb/sda/e2e file
2. Scan group (usually OCT + enface, there may be just one of these per record, but sometimes multiple scan-groups are captured in a single record in which case they are given different group ids)
3. Scan (in the privateeye/crystaleye this is represented by each 'image' entry)
4. Image (usually 1 per scan, except for OCT scans whre there are multiple)


Scan / scan grp (volume) / scan image
Scan metadata
Mask (single image)
Mask volume
Fovea (prediction)
Registration projection
Classification output
Dataset
Model

"""

from abc import ABCMeta, abstractmethod, abstractproperty
from cached_property import cached_property

from collections import defaultdict
import json
import os

class MetadataRecord():
    """
    Separate holder object for metadata information
    Good to keep separate to avoid data-deduplication
    Can also add getters for common pieces of info to make transparent to format
    """
    def __init__(self, data, location=None):
        self._format = None
        self._data = data
        self._location = location
        
    @property
    def raw(self):
        return self._data
    
    @property
    def location(self):
        return os.path.dirname(os.path.abspath(self._location))
    
    def get_view(self, format=None):
        return MetadataViewCrystalEye(self)
    
    @classmethod
    def load(cls, file_path):
        if not file_path.endswith(".json"):
            file_path = os.path.join(file_path, "metadata.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
        return MetadataRecord(data, file_path)
    
        
class MetadataView(metaclass=ABCMeta):
    def __init__(self, record, view_info=None):
        self._record = record
        self._view_info = view_info if view_info else dict()
        
    def get_by_path(self, path, root=None):
        if root:
            pos = root
        else:
            pos = self._record.raw
        for index in path:
            pos = pos[index]
        return pos
    
    def get_view(self, view_info):
        return self.__class__(self._record, view_info)
    
    @property
    def scan_number(self):
        return self._view_info['scan_number']
    
    @property
    def image_number(self):
        return self._view_info['image_number']
    
    @property
    def bscan_index(self):
        return self.image_number
    
    
    @abstractproperty
    def _scan_level(self):
        raise NotImplementedError()
        
    @property
    def _scan_root(self):
        return self._scan_level[self.scan_number]
    
    @property
    def n_scans(self):
        return len(self._scan_level)
    
    @property
    def scans(self):
        return [ self.get_view({'scan_number': i }) for i in range(self.n_scans) ]
    
    
    @abstractproperty
    def _image_level(self):
        raise NotImplementedError()
        
    @property
    def _image_root(self):
        return self._image_level[self.image_number]
        
    @property
    def n_images(self):
        return len(self._image_level)
        
    @property
    def images(self):
        return [ self.get_view({'scan_number': self.scan_number, 'image_number': i }) for i in range(self.n_images) ]
    
    @property
    def bscans(self):
        return self.images
    
    def get_groups(self):
        groups = defaultdict(list)
        for scan in self.scans:
            groups[scan.group].append(scan)
        return list(groups.values())
    
    
    @abstractproperty
    def group(self):
        raise NotImplementedError()
        
    @abstractproperty
    def source_id(self):
        raise NotImplementedError()
         
    @abstractproperty
    def modality(self):
        raise NotImplementedError()
        
    @abstractproperty
    def location(self):
        raise NotImplementedError()
               
        
class MetadataViewCrystalEye(MetadataView):
    
    # TODO: Make a way to automatically register these paths
    
    @cached_property
    def manufacturer(self):
        return self.get_by_path(["exam", "manufacturer"])
    
    @cached_property
    def _scan_level(self):
        return self.get_by_path(["images", "images"])
    
    @cached_property
    def _image_level(self):
        return self.get_by_path(["images", "images", self.scan_number, "contents"])
                                 
    @cached_property
    def group(self):
        return self.get_by_path(["group"], root=self._scan_root)
    
    @cached_property
    def source_id(self):
        return self.get_by_path(["source_id"], root=self._scan_root)
    
    @cached_property
    def modality(self):
        return self.get_by_path(["modality"], root=self._scan_root)
        
    @cached_property
    def location(self):
        scan_idx = self.bscan_index if 'OCT' in self.modality else 0
        file_name = "{}_{}.png".format(self.source_id, scan_idx)
        return os.path.join(self._record.location, file_name)

        
class Record():
    """
    Datastructure for holding information about the specific "record" of a scan
    Mianly to keep track of it for saving purposes
    """
    def __init__(self):
        self._file_path = None
        self._patient_id = None
        self._scan_id = None
        
        self._metadata = None # Keep reference to metadata?
        
        self._entries = None # Keep reference to entries?
