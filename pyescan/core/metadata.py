from abc import ABCMeta, abstractclassmethod, abstractmethod, abstractproperty
from cached_property import cached_property

from collections import defaultdict
import os

class MetadataRecord():
    """
    Separate holder object for metadata information
    Good to keep separate to avoid data-deduplication
    Can also add getters for common pieces of info to make transparent to format
    """
    def __init__(self, data, location=None, parser=None):
        self._format = None
        self._data = data
        self._location = location
        self._parser = parser # Should maybe call this 
        
    @property
    def raw(self):
        return self._data
    
    @property
    def location(self):
        return os.path.dirname(os.path.abspath(self._location))
    
    def get_view(self, format=None, parser=None):
        return MetadataView(self, parser=parser)
    
    @classmethod
    def load(cls, file_path):
        """
        Class method for loading - should maybe alter this to auto-detect loading in future
        """
        raise NotImplementedError()
    
        
class MetadataView():
    def __init__(self, record, view_info=None, parser=None):
        self._record = record
        self._view_info = view_info if view_info else dict()
        self._parser = parser
    
    @property
    def parser(self):
        return self._parser or self._record._parser
    
    def __getattribute__(self, attribute_name):
        try:
            return super().__getattribute__(attribute_name)
        except AttributeError as e:
            # Try get value from parser
            res = self.get_value(attribute_name)
            if not res is None:
                return res
            else:
                # If not found, raise the standard AttributeError
                raise 
            
    def get_view(self, view_info):
        return self.__class__(self._record, view_info=view_info, parser=self._parser)
    
    def get_value(self, attribute_name):
        return self.parser.get_value(attribute_name, self._record, self._view_info)
    
    @property
    def scan_number(self):
        return self._view_info['scan_number']
    
    @property
    def image_number(self):
        return self._view_info['image_number']
    
    @property
    def bscan_index(self):
        return self.image_number
    
    @property
    def scans(self):
        return [ self.get_view({'scan_number': i }) for i in range(self.n_scans) ]
        
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
    
    
class MetadataParser(metaclass=ABCMeta):
    @abstractmethod
    def get_value(attribute_name, metadata_record, view_info):
        return NotImplementedError()

class MetadataParserJSON(MetadataParser):
    
    # TODO: Make a way to automatically register these paths
    def _get_by_path(self, metadata_record, path, root=None):
        pos = root or metadata_record.raw
        for index in path:
            pos = pos[index]
        return pos

    def _get_path(self, attribute_name, view_info):
        return self._path_map.get(attribute_name, None)
    
    def _map_path(self, path, view_info):
        mapped_path = list()
        for p in path:
            if p.startswith('{') and p.endswith('}'):
                p = view_info[p.strip('{}')]
            mapped_path.append(p)
        return mapped_path
        
    def get_value(self, attribute_name, metadata_record, view_info):
        if attribute_name in self._overrides:
            return self._overrides[attribute_name](metadata_record, view_info)
        
        path = self._get_path(attribute_name, view_info)
        if path is None:
            return None
    
        mapped_path = self._map_path(path, view_info)
        return self._get_by_path(metadata_record, mapped_path)
    
class MetadataParserCSV(MetadataParser):
    @abstractmethod
    def _get_records_subset(self, metadata_record, view_info):
        return NotImplementedError()
    
    def get_value(self, attribute_name, metadata_record, view_info):
        if attribute_name in self._overrides:
            return self._overrides[attribute_name](metadata_record, view_info)
        
        records_subset = self._get_records_subset(metadata_record, view_info)
        
        col_name = self._col_map.get(attribute_name)
        if col_name is None: return None
        return records_subset[col_name].values[0]
    