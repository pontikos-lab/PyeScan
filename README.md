# PyeScan

PyeScan is a python library for streamlining the process of working with retinal scans in python and provide a common interface for various processes (e.g. running various models). The idea is for the main libary to develop core, implementation-agnostic, fucntionality with various helpers/loaders/savers/views etc to deal with specific formats.


## Installing

Installation is pretty standard
```
git clone https://github.com/Moorfields-Reading-Centre/PyeScan.git
cd PyeScan/
pip install -e .
```

N.B: Dependencies aren't fully set up yet (so you may have to isntall more packages with `pip` as needed).

Otherwise you can just run things in the root level folder of the repo if you want to just test things out.


## Example usage

You can use `load_record_from_CE` to load a set of scans from a single PrivateEye/CrystalEye export 
(N.B: This is currently only tested on Heidelberg scans):
```python
from pyescan.CELoader import load_record_from_CE

record_path = "/[PATH_TO_DATA]/[XXXXXXXX.pat]/[XXXXXXXX.sdb]/"
scans = load_record_from_CE(record_path)
```

`scans` is a list of scan instances of the various scan classes (all of which inberit from the `BaseScan`)

Alternatively you can load from s dataframe (restricted to a single `sdb`):

```python
from pyescan.CELoader import load_record_from_df

import pandas as pd
df = pd.read_csv("[PATH_to_CSV]") #e.g. generated by get_pe_export_summary

df_scan = df.query("sdb == '01437803.sdb'")
scans = load_record_from_df(df_scan)
```

This is "configurable" to work with any data format, see `CrystalEyeParser` and `CrystalEyeParserCSV` in `CELoader.py` to see an example of how this is set up (N.B: These WILL be updated in future, hopefully to allow for proper configuration).

Data loading is done in a 'lazy' fashion where the actual images are only loaded when they are needed (however you can force load with `scan.preload()` if needed) meaning you can load scans for basic operations without having to load all the data - only the metadata is loaded.

You can get the raw data within the scan by calling `scan.image`, which will return a wrapper class for the image, or `scan.data` which will return a raw numpy array of the image. For OCT scans you can call `scan.enface` for the enface scan and `scan.bscans` to return the bscan array.

The library provides support for advanced indexing and slicing of the scans, for easy accessing of a certain region, e.g. `scan.bscans[:n,:100,:100]` (or `scan.data[:,:100,:100]` will return a numpy array of an `nx100x100` of the OCT volume.

You can also view a scan inside a jupyter notebook with `display(scan)` (or simply `scan` at the end of a cell), with support for interactive viewer for OCT.

You can access information through `scan.metadata.QUERY` which can be used to get various information about this scan. Only a few functiosn are currently implemented but these should be added by adding the required property to the relevant `MetadataView` class (in this case `MetadataViewCrystalEye`). This is hierarchical so scan/image specific information can be implemented in the `MetadataView` class, and then accessed seamlessly by the scan objext.


### Annotations

Annotations support is still in development, but can be loaded from file and added to scan:

```python
# This loads a scan from a dataframe as long as the right columns are set
# The default mapping is: (this can be changed by supplying a dict as seen below)
#    {"group": "group", "source_id": "source_id", "modality": "modality", "image_location": "file_path", "n_images": "number_of_images" }
from pyescan.CELoader import load_record_from_df
scan, *_ = load_record_from_df(df_scan, {'image_location': 'file_path_original'})

# This loads a scan from a dataframe, this one only needs file_path_col, bscan_index_col, and optionally feature_col
# If feature_col is supplied then it will automatically create a dict of all features, otherwise it will just turn a single (unnamed) annotation
from pyescan.annotation_loader import load_annotation_from_df
annotations = load_annotation_from_df(df_scan.query("modality == 'OCT'"), feature_col='feature')

# Add the loaded annotations to the scan
#  This will probably be update in future
scan.add_annotations(annotations)

# Then display the scan (can sometimes take a while as it needs to load all the images)
scan
```

Loading can also be done using `load_annotation_from_folder` which works in the same way but first indexes the folder structure according to a specified pattern using `summarise_dataset` and uses the resulting dataframe as an intermediate.


### Unified Metrics system

`pyescan.metrics` contains a bunch of useful metrics which can be run to get various statistics. This is built into a centralised dependency system so that any intermediate requirments are automatically computed (and cached). These used to all be set up to run directly on dataframes via a dataframe row, howeveer this was inflexible, and not really compatible with the idea of pyescan.

Example running on a dataframe:
```python
from pyescan.metrics.registry import PYESCAN_GLOBAL_METRICS
print("Loaded metrics:")
print("\n\n".join([str(metric) for metric in PYESCAN_GLOBAL_METRICS]))

from pyescan.metrics.processor import MetricProcessor
from pyescan.metrics.helpers import PandasRowWrapperHelper
from tqdm.notebook import tqdm

import pandas as pd
df = pd.read_csv("EXAMPLE_CSV.csv")

processor = MetricProcessor(PYESCAN_GLOBAL_METRICS.metrics)
metric, params = processor.get_metric_by_stat("pixel_count_oct_4.0mm_superior")
print("Running", metric.name, "with params", params)
    
results = []
traces = []
for idx, row in tqdm(df.iterrows()):
    col_map = {"file_path_mask": "file_path", "scan_width_px": "size_width", "scan_height_px": "size_height"}

    wrapped = PandasRowWrapperHelper(row, col_map)
    result = processor._process_metric(wrapped, metric, params)
    
    traces.append(result['computed_metrics'])
    results.append(result['computed_stats'])
    
df_stats = pd.DataFrame(results)
df_stats```

New metrics can also be added - there is a function wrapper provided for doing this, see `pyescan.metrics.metrics.py` for examples (documentation to follow).


### Helper functions

In `pyescan.tools.dataset_utils` there are two useful dataset functions `summarise_dataset`, and `get_pe_export_summary`

`summarise_dataset` simply traverses a directory structure and finds all files matching a particular file structure, and summarises them in a single pandas dataframe.

For example
```
summarise_dataset([ROOT_DIRECTORY], structure="{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png"
```
Will get all files that look like `{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png` (compared to the `[ROOT_DIRECTORY]`) and list them in a table with columns for `pat`, `sdb`, `source_id`, and `bscan_index` (as well as `file_path`). As you cna imagine this can be very useful for loading different types of input file structures.

The other function, `get_pe_export_summary`, is similar, but traverses through a Private/CrystalEye export and dives into the metadata, producing a dataframe with a full summary of all scans/images including images (there is some data still missing, in particular the image-level metadata ~~but the plan is to add this at some point~~ is now included, but some fields are currently omitted for brevity). This is useful to make metadata summary CSVs which can be merged with other datasets (e.g. annotations/masks) as needed 

There are also a number of functions broken out as command line tools:
`summarise_pe_export`,`summarise_dataset`, `run_function_on_csv`, `run_function_over_csv`, `narrow_to_wide`
Use `[CMD] --help` to see how to use each one.


# Explanation + Development

Currently PyeScan is just about laying the groundwork in order to make things easier, but the idea is that we should be able to eventually port over functionality from existing code (and it should be much nicer / more streamlined to implement). Whenever you find a comon task you have to do constantly, consider adding it to the library!

The core object of PyeScan is the scan object which inherits from the `BaseScan` class (though some scan objects are also compositions of other BaseScan-derived instances, as well as being a `BaseScan` in their own right).

These scan classes provide a standard interface for various operations, along with various helper functionality.

A big part of how this is achieved is with `MetadataView` objects which (as the name suggests) provide a 'view' onto the raw metadata, translating from the original metadata structure (by prividing various acessor properties), while also handling managing which part of the metadata the particular scan refers to.

Loading of actual images is currently done 'lazily' so data is only loaded when actually needed. This is intended to be largely seamless for the end user, but obviously there may be times where the user wants to 'precache' data, so `preload` and `unload` functions are included for forcing loading/unloading.

Currently only loading images directly from disk is supported, but it should be possible to implement other ways of loading (e.g. directly from e2e file or whatever) fairly transparently.

Annotations/masks can be loaded and added to scans. In future there will be a system to properly identify which mdoel and which feature the annotation came from/refers to, and ways of automatically searching for annotations and linking them to certain scans.

During development of PyeScan it was found there was still a lot of working with dataframes, rather than loading in scans as Pyescan objects explicitly. The idea with the metric system is then to make it easy to work with either depending on the user needs.


## TODO

Main items:
- Proper documentation (+ Linting)
- Proper tests
- Develop+test running metrics on PyeScan scan objects
- Type annotations/hints
- Rendering of annotations on enface with contours/cloropleth for thickness
- ~~Annotations/masks~~ (Partially) Done!
- Port over functionality from PEExportBase
