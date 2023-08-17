def flatten_dict(dict_in, name=""):
    # Recursive function that "flattens" a nested dict by concatenating key names of sub-dicts
    # e.g. { "a": 1, "b": { "c": 2, "d": { "e": 3 } } } becomes { "a": 1, "b_c": 2, "b_d_e": 3}
    # Note: Does not handle lists.

    dict_out = dict()
    if type(dict_in) is dict:
        # Recursive case
        for k, v in dict_in.items():
            dict_out.update(flatten_dict(v, name="{}_{}".format(name,k) if name else k))
    else:
        dict_out[name] = dict_in
    return dict_out

# TODO: Get the bscan-level 
def get_pe_export_summary(scans_folder, file_structure="pat/sdb", merged=True,  metadata_out=None, file_list_out=None):
    import json
    import os
    import pandas as pd
    import tqdm
        
    scan_records = list()
    file_records = list()

    for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(scans_folder)):
        # Only check for leaf directories
        if not len(dirnames) == 0: continue

        if not 'metadata.json' in filenames: continue

        file_structure_keys = file_structure.split("/")
        file_structure_elems = dirpath.split('/')[-len(file_structure_keys):]
        file_structure_dict = dict(zip(file_structure_keys, file_structure_elems))

        UID_path = "/".join(file_structure_elems)

        with open(os.path.join(dirpath, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

            scan_date = metadata['exam']['scan_datetime']
            series_info = flatten_dict(metadata['series'])

            images = metadata['images']['images']

            for i, image in enumerate(images):
                image_data = dict()
                image_data.update(file_structure_dict)

                image_data['scan_uid'] = UID_path + "/" + image['source_id']

                for attr in ['source_id', 'group', 'modality', 'field_of_view']:
                    image_data[attr] = image[attr]
                image_data['scan_number'] = i 
                image_data['date'] = scan_date
                image_data.update(series_info)
                
                # workaround for name collision
                image_data['series_source_id'] = metadata['series']['source_id']
                image_data['source_id'] = image['source_id']
                
                image_data['number_of_images'] = len(image['contents'])

                for attr in ['size', 'dimensions_mm', 'resolutions_mm']:
                    image_data.update(flatten_dict(image[attr], attr))

                scan_records.append(image_data)

        for filename in filenames:

            # Assume filename is of the form [SOURCE-ID]_[bscan_index].png
            if not ".png" in filename: continue

            source_id, bscan_index = filename.rsplit("_", 1)
            bscan_index = bscan_index.rsplit(".",1)[0] # remove file_extension
            filepath = os.path.join(dirpath, filename)

            file_records.append({"file_path": filepath,
                                 "file_name": filename,
                                 "scan_uid": UID_path + "/" + source_id,
                                 "bscan_index": bscan_index })

    df_metadata = pd.DataFrame(scan_records)
    df_files = pd.DataFrame(file_records)

    #metadata_out = metadata_out if metadata_out else os.path.join(scans_folder,"scan_metadata.csv")
    #file_list_out = file_list_out if file_list_out else os.path.join(scans_folder,"file_list.csv")
    
    if metadata_out:
        df_metadata.to_csv(metadata_out, index=False)
    if file_list_out:
        df_files.to_csv(file_list_out, index=False)
        
    if merged:
        df_files = df_files.merge(df_metadata, how='left', on='scan_uid')
        return df_files
    else:
        return df_files, df_metadata

def get_median_bscans(df, scan_id_key='scan_uid', index_key='bscan_index'):
    median_row_ids = list()
    for scan_id, df_scan in df.groupby(scan_id_key):
        scan_indices = sorted(df_scan[index_key].values)
        middle_slice = scan_indices[len(scan_indices) // 2]
        rows = df_scan[df_scan[index_key] == middle_slice].index
        median_row_ids.extend(rows)

    df_middle = df.loc[median_row_ids]
    return df_middle


def structure_to_regex(structure_pattern):
    # TODO: Add automatic conversion of python formatting strings to regex
    # TODO: Addoptional way to specify wildcards for path
    import re
    
    # Escape characters not between {}
    escaped_structure = re.sub(
        r"(?<!{)([^{}]*)*(?![^{}]*})", # Need to add the [^{}]* to the lookahead
        lambda match: re.escape(match.group(0)),
        structure_pattern,
    )
    
    """
    This code uses the re.sub() method to replace each instance of a curly brace 
    expression with a corresponding regular expression pattern. The pattern uses
    a named capture group to capture the value of the component, and it includes
    a non-capturing group that matches either the specified regular expression
    pattern (if provided) or any sequence of non-slash characters (if no pattern 
    is specified). (Generated by Chat-GPT)
    """
    
    # Replace items between brackets with groups
    regex_pattern = re.sub(
        r"{([a-zA-Z0-9_]*?)(?::(.*?))?}",
        #r"{(.*?)(?::(.*?))?}",
        lambda match: f"(?P<{match.group(1)}>{match.group(2) or '[^/]*?'})",
        escaped_structure,
    )
    
    return regex_pattern
    

def summarise_dataset(dataset_root, structure="{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png", regex=None):
    import os
    import re
    import pandas as pd
    import tqdm
    
    dataset_root = os.path.abspath(dataset_root)
    
    regex_pattern = regex if regex else structure_to_regex(structure)
    
    records = list()
    for root, dirs, filenames in tqdm.tqdm(os.walk(dataset_root)):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, dataset_root)
            
            match = re.match(regex_pattern, rel_path)
            if match:
                record = { "file_path": file_path, "file_path_relative": rel_path }
                record.update(match.groupdict())
                records.append(record)
    return pd.DataFrame(records)