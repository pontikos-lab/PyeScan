from pyescan.core.metadata import MetadataRecord, MetadataParserJSON, MetadataParserCSV
from pyescan.core.scan_building import build_from_metadata

import os

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


class CrystalEyeParser(MetadataParserJSON):
    _scan_level = ["images", "images", "{scan_number}"]
    _image_level = _scan_level + ["contents", "{image_number}"]
    
    _path_map = {
        "group": _scan_level + ["group"],
        "source_id": _scan_level + ["source_id"],
        "modality": _scan_level + ["modality"],
        "manufacturer": _scan_level + ["manufacturer"]
    }
    
    def __init__(self):
        self._overrides = {
            "n_scans": self.n_scans,
            "n_images": self.n_images,
            "image_location": self.image_location
        }
    
    def n_scans(self, metadata_record, view_info):
        path = self._map_path(self._scan_level[:-1], view_info)
        return len(self._get_by_path(metadata_record, path))
    
    def n_images(self, metadata_record, view_info):
        path = self._map_path(self._image_level[:-1], view_info)
        return len(self._get_by_path(metadata_record, path))

    def image_location(self, metadata_record, view_info):
        modality = self.get_value('modality', metadata_record, view_info)
        bscan_index = view_info['image_number'] if 'OCT' in modality else 0
        source_id = self.get_value('source_id', metadata_record, view_info)
        
        file_name = "{}_{}.png".format(source_id, bscan_index)
        return os.path.join(metadata_record.location, file_name)

    
class CrystalEyeParserCSV(MetadataParserCSV):
    _base_col_map = {
        "group": "group",
        "source_id": "source_id",
        "modality": "modality",
        "image_location": "file_path",
        "n_images": "number_of_images"
    }
    
    def __init__(self, column_headings={}):
        self._col_map = self._base_col_map.copy()
        self._col_map.update(column_headings)
        self._overrides = { "n_scans": self.n_scans }
        
    def _get_records_subset(self, metadata_record, view_info):
        df = metadata_record.raw
        if "scan_number" in view_info:
            scan_number = metadata_record.raw.source_id.unique()[view_info["scan_number"]]
            df = df.query("source_id == @scan_number")
        if "image_number" in view_info:
            image_number = view_info["image_number"]
            df = df.query("bscan_index == @image_number")
        return df
    
    def n_scans(self, metadata_record, view_info):
        #records_subset = self._get_records_subset(metadata_record, view_info)
        return metadata_record.raw.source_id.nunique()
    

def load_record_from_CE(path_to_record_folder, format=None):
    
    import json
    file_path = path_to_record_folder
    if not file_path.endswith(".json"):
        file_path = os.path.join(file_path, "metadata.json")
    with open(file_path, 'r') as f:
        json_data = json.load(f)
        
    record = MetadataRecord(json_data, file_path)
    metadata = record.get_view(parser=CrystalEyeParser())
    
    return build_from_metadata(metadata)


def load_record_from_df(df_scan, column_headings={}):
    record = MetadataRecord(df_scan)
    metadata = record.get_view(parser=CrystalEyeParserCSV(column_headings=column_headings))
            
    return build_from_metadata(metadata)