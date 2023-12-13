from pyescan.core.metadata import MetadataRecord

from pyescan.core.image import LazyImage

from pyescan.core.scan_enface import IRScan, FAFScan
from pyescan.core.scan_oct import BScan, BScanArray, OCTScan

#TODO: Move more of the building of the actual classes elsewhere
def load_record_from_CE(path_to_record_folder, format=None):

    record = MetadataRecord.load(path_to_record_folder)
    metadata = record.get_view()
    
    scans = list()
    for group in metadata.get_groups():
        modalities = [scan.modality for scan in group]
        
        if modalities == ['SLO - Infrared', 'OCT']:
            ir_meta, oct_meta = group
            
            ir_img_path = ir_meta.location
            ir_img = LazyImage(ir_img_path)
            enface = IRScan(image=ir_img, metadata=ir_meta)
            
            bscans = list()
            for i, bscan_meta in enumerate(oct_meta.bscans):
                bscan_img_path = bscan_meta.location
                bscan_img = LazyImage(bscan_img_path)
                bscan = BScan(bscan_img, i, bscan_meta)
                bscans.append(bscan)
            bscan_array = BScanArray(bscans)
            scan = OCTScan(enface, bscan_array, metadata=oct_meta)
            scans.append(scan)
            
        elif modalities == ['AF - Blue']:
            scan_meta = group[0]
            scan_img_path = scan_meta.location
            scan_img = LazyImage(scan_img_path)
            scan = FAFScan(image=scan_img, metadata=scan_meta)
            scans.append(scan)
            
        else:
            print("Skipping Group", modalities)
            
    return scans