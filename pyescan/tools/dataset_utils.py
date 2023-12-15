import click

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
def get_pe_export_summary(export_location, file_structure="pat/sdb", merged=True, skip_image_level=False):
    import json
    import os
    import pandas as pd
    import tqdm
        
    scan_records = list()
    file_records = list()

    pbar = tqdm.tqdm(os.walk(export_location))
    for dirpath, dirnames, filenames in pbar:
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

            scans = metadata['images']['images']

            for i, scan in enumerate(scans):
                scan_data = dict()
                scan_data.update(file_structure_dict)

                scan_data['scan_uid'] = UID_path + "/" + scan['source_id']

                for attr in ['source_id', 'group', 'modality', 'field_of_view']:
                    scan_data[attr] = scan[attr]
                scan_data['scan_number'] = i 
                scan_data['date'] = scan_date
                scan_data.update(series_info)
                
                # workaround for name collision
                scan_data['series_source_id'] = metadata['series']['source_id']
                scan_data['source_id'] = scan['source_id']
                
                scan_data['number_of_images'] = len(scan['contents'])

                for attr in ['size', 'dimensions_mm', 'resolutions_mm']:
                    scan_data.update(flatten_dict(scan[attr], attr))
                    
                if skip_image_level:
                    scan_records.append(scan_data)
                else:
                    images = scan['contents']
                    for j, image in enumerate(images):
                        image_data = dict()
                        image_data.update(scan_data)
                        
                        #image_data['image_number'] = j
                        image_data['bscan_index'] = str(j)
                        
                        image_data['image_capture_datetime'] = image.get('capture_datetime')
                        image_data['image_quality_heidelberg'] = image.get('quality')
                        
                        if image.get('photo_locations'):
                            locations = image['photo_locations'][0]
                            image_data.update(flatten_dict(locations, 'bscan_location'))
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
            
        pbar.set_postfix({'scans_found': len(scan_records)})
            
    df_metadata = pd.DataFrame(scan_records)
    df_files = pd.DataFrame(file_records)
        
    if merged:
        merge_cols = ['scan_uid']
        if not skip_image_level: merge_cols = merge_cols + ['bscan_index']
        df_files = df_files.merge(df_metadata, how='left', on=merge_cols)
        return df_files
    else:
        return df_files, df_metadata
    
@click.command()
@click.argument('export_location', type=click.Path(exists=True), required=True)
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--file_structure', type=str, default='pat/sdb', help='File structure gias list of names separated by / (default: pat/sdb).')
@click.option('--skip_image_level/--include-image-level', default=False, help='Skip image level information for faster parsing (default: False).')
def get_pe_export_summary_cli(export_location, output_csv, file_structure="pat/sdb", merged=True, skip_image_level=False):
    df = get_pe_export_summary(
        export_location,
        file_structure=file_structure,
        skip_image_level=skip_image_level
    )
    
    if output_csv:
        # If output_location is provided, save the DataFrame to the specified location
        df.to_csv(output_csv, index=False)
        click.echo(f"Result saved to {output_location}")
    else:
        # If output_location is not provided, save to scans_folder with a default filename
        default_output = os.path.join(scans_folder, "metadata_summary.csv")
        df.to_csv(default_output, index=False)


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
    pbar = tqdm.tqdm(os.walk(dataset_root))
    for root, dirs, filenames in pbar:
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, dataset_root)
            
            match = re.match(regex_pattern, rel_path)
            if match:
                record = { "file_path": file_path, "file_path_relative": rel_path }
                record.update(match.groupdict())
                records.append(record)
                
        pbar.set_postfix({'scans_found': len(records)})
        
    return pd.DataFrame(records)


@click.command()
@click.argument('target_location', type=click.Path(exists=True), required=True)
@click.argument('output_csv', type=click.Path(), required=True)
@click.option('--format', type=str, default='{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png',
              help='File structure gias list of names separated by / (default: {pat}/{sdb}/{source_id}_{bscan_index:\d+}.png).')
def summarise_dataset_cli(target_location, output_csv, format):
    df = summarise_dataset(target_location, structure=format)
    df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")

    
def get_median_bscans(df, scan_id_key=['sdb', 'source_id'], index_key='bscan_index'):
    median_row_ids = list()
    for scan_id, df_scan in df.groupby(scan_id_key):
        scan_indices = sorted(df_scan[index_key].values)
        middle_slice = scan_indices[len(scan_indices) // 2]
        rows = df_scan[df_scan[index_key] == middle_slice].index
        median_row_ids.extend(rows)

    df_middle = df.loc[median_row_ids]
    return df_middle


@click.command()
@click.argument('function', type=str)
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def run_function_on_csv_cli(csv_file, function, output_csv=None, skip_prompt=False):
    """
    Runs the specified function on the specified CSV and saves to the target location.
    If output_csv is not specified this will default to overwriting the input CSV.
    Function should map from Pandas DataFrame to Dataframe.
    """
    if not output_csv:
        confirmation = skip_prompt or click.confirm(f'No output location specified. Are you sure you want to overwrite {csv_file}?')
        if confirmation:
            output_csv = csv_file
        else:
            click.echo("Operation aborted.")
            return

    # Check if the specified function is available
    try:
        function_to_apply = globals()[function]
    except KeyError:
        raise ValueError(f"Function '{function}' not found in the global scope.")
        
    import pandas as pd
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Apply function
    result_df = function_to_apply(df)
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")


def run_function_on_dataframe(df, fn, column, threaded=True):
    import tqdm

    if threaded:
        from pathos.multiprocessing import Pool
        with Pool(64) as p:
            df[column] = list(tqdm.tqdm(p.imap(fn, list(df.itertuples(index=False))), total=len(df)))
    else:
        outputs = list()
        for row in tqdm.tqdm(df.itertuples(index=False), total=len(df)):
            outputs.append(fn(row))
        df[column] = outputs
    return df
    
@click.command()
@click.argument('function', type=str)
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path(), default=None, required=False)
@click.option('--column_headings', '-c', type=click.STRING, help='Column headings separated by commas.')
@click.option('--threaded', is_flag=True, help='Use threading (recommended, requires pathos).')
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def run_function_over_csv_cli(csv_file, function, output_csv=None, column_headings=None, threaded=None, skip_prompt=True):
    """
    Runs the specified function on each row of the specified CSV and appends the result as additional columns, with the names given in column_headings.
    If output_csv is not specified this will default to overwriting the input CSV.
    Function should accept a row parameter, where columns can be accessed by row.column_name, and output a single value / tuple of values.
    """
    if not output_csv:
        confirmation = skip_prompt or click.confirm(f'No output location specified. Are you sure you want to overwrite {csv_file}?')
        if confirmation:
            output_csv = csv_file
        else:
            click.echo("Operation aborted.")
            return

    # Check if the specified function is available
    function_split = function.rsplit(".",1)
    if len(function_split) == 1:
        function_name = function_split
        # Try get the function from globals()
        if function in globals():
            function_to_apply = globals()[function]
        else:
            click.echo(f"ERROR: Function '{function}' not found in the global scope.")
            return
    else:
        module_name, function_name = function_split
        # Try import function from module
        try:
            import importlib
            imported_module = importlib.import_module(module_name)
            function_to_apply = getattr(imported_module, function_name)
        except Exception as e:
            click.echo(f"ERROR: Failed to import {function_name} from {module_name}:")
            raise e
        
    # Deafult to using the function name
    if not column_headings:
        column_headings = function
        
    # Get headings
    column_headings = column_headings.split(",")
    if len(column_headings) == 1: column_headings = column_headings[0]
        
    import pandas as pd
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Run function over dataframe
    result_df = run_function_on_dataframe(df, function_to_apply, column_headings, threaded)
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv, index=False)
    click.echo(f"Result saved to {output_csv}")
    
    
def detect_pivot_cols(df, pivot_col, identifier_cols):
    # Find columns with identical values for each unique combination in the target columns (from ChatGPT)
    index_cols = list()
    for col in df.columns:
        if col in identifier_cols: continue # skip
        if (df.groupby(identifier_cols)[col].nunique() <= 1).all():
            index_cols.append(col)
    index_cols = identifier_cols + index_cols

    # Get remaining cols
    value_cols = list(set(df.columns) - set(index_cols + [pivot_col]))
    return index_cols, value_cols

    
def narrow_to_wide(df, pivot_col, identifier_cols, verbose=True, flatten_column_names=True):
    # Get pivot cols
    index_cols, value_cols = detect_pivot_cols(df, pivot_col, identifier_cols)
    
    if verbose:
        print("Pivoting on ", value_cols)

    df_summary = df.pivot(index=index_cols, columns=pivot_col, values=value_cols).reset_index()
    
    if flatten_column_names:
        # Rename columns, needs assigning to intermediate variable or weird stuff happens
        columns = [ "_".join([str(c) for c in col if c]) for col in df_summary.columns]
        df_summary.columns = columns
    
    return df_summary


@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.argument('pivot_col', type=str)
@click.argument('identifier_cols', type=str)
@click.option('--flatten_column_names', is_flag=True, default=True, help='Flatten names of column headings.')
@click.option('--skip_prompt', '-y', is_flag=True, help='Skip confirmation.')
def narrow_to_wide_cli(csv_file, output_csv, pivot_col, identifier_cols, flatten_column_names=True, skip_prompt=False):
    """
    Pivots the specified CSV around the column heading given by pivot_col and ouputs new CSV to output_csv.
    Automatically detects which columns to pivot based on identifier_cols.
    identifier_cols should be supplied as a column separated list of a minimum number of required columns to identify each element to be pivoted.
    """
    
    import pandas as pd
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Get pivot cols
    identifier_cols = identifier_cols.split(",")
    index_cols, value_cols = detect_pivot_cols(df, pivot_col, identifier_cols)
    
    
    if not (skip_prompt or click.confirm(f'The following columns will be replicated across {pivot_col} values:\n{value_cols}\n\n Continue?')):
        click.echo("Operation aborted.")
        return

    # Pivot
    result_df = df.pivot(index=index_cols, columns=pivot_col, values=value_cols).reset_index()
    
    if flatten_column_names:
        # Rename columns, needs assigning to intermediate variable or weird stuff happens
        columns = [ "_".join([str(c) for c in col if c]) for col in result_df.columns]
        result_df.columns = columns
    
    # Save the result to a CSV file
    result_df.to_csv(output_csv)
    click.echo(f"Result saved to {output_csv}")

