from pyescan.core.annotation import MaskImage, MaskVolume, AnnotationOCT

def _build_annotation_from_file_paths(file_paths):
    masks = list()
    for i, file_path in enumerate(file_paths):
        #mask_img = MaskImage(file_path)#None if i==3 else file_path)
        mask_img = MaskImage(None if i==3 else file_path)
        masks.append(mask_img)
    mask_array = MaskVolume(masks)
    annotation = AnnotationOCT(masks=mask_array)
    return annotation

def _build_annotation_from_dataframe_base(df, file_path_col='file_path', index_col='bscan_index'):
    df_copy = df.copy()
    
    # Convert bscan_index to int (it's float after to_numeric)
    df[index_col] = df[index_col].astype(int)
    
    # Ensure the DataFrame is sorted by bscan_index
    df_sorted = df.sort_values(index_col)
    max_index = df_sorted[index_col].max()
    
    # Create an array of None values with length max_index + 1
    file_paths = [None] * (int(max_index) + 1)
    
    # Fill in the array with file paths where bscan_index matches
    for _, row in df_sorted.iterrows():
        file_paths[row[index_col]] = row[file_path_col]
    return _build_annotation_from_file_paths(file_paths)

def load_annotation_from_df(df, file_path_col='file_path', index_col='bscan_index', feature_col=None):
    if feature_col:
        annotations_dict = {}
        for feature, df_feat in df.groupby(feature_col):
            ann = _build_annotation_from_dataframe_base(df_feat, file_path_col, index_col)
            annotations_dict[feature] = ann
        return annotations_dict
    else:
        ann = _build_annotation_from_dataframe_base(df, file_path_col, index_col)
        return ann