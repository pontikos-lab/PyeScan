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

Otherwise you can just run things in the root level folder of the repo to just test things out.


## Example usage

You can use `load_record_from_CE` to load a set of scans from a single PrivateEye/CrystalEye export 
(N.B: This is currently only tested on Heidelberg scans):
```
from pyescan.CELoader import load_record_from_CE

record_path = "/[PATH_TO_DATA]/[XXXXXXXX.pat]/[XXXXXXXX.sdb]/"
scans = load_record_from_CE(record_path)
```

`scans` is a list of scan instances of the various scan classes (all of which inberit from the `BaseScan`)

Data loading is done in a 'lazy' fashion where the actual images are only loaded when they are needed (however you can force load with `scan.preload()` if needed) meaning you can load scans for basic operations without having to load all the data - only the metadata is loaded.

You can get the raw data within the scan by calling `scan.image`, which will return a wrapper class for the image, or `scan.data` which will return a raw numpy array of the image. For OCT scans you can call `scan.enface` for the enface scan and `scan.bscans` to return the bscan array.

The library provides support for advanced indexing and slicing of the scans, for easy accessing of a certain region, e.g. `scan.bscans[:n,:100,:100]` (or `scan.data[:,:100,:100]` will return a numpy array of an `nx100x100` of the OCT volume.

You can access information through `scan.metadata.QUERY` which can be used to get various information about this scan. Only a few functiosn are currently implemented but these should be added by adding the required property to the relevant `MetadataView` class (in this case `MetadataViewCrystalEye`). This is hierarchical so scan/image specific information can be implemented in the `MetadataView` class, and then accessed seamlessly by the scan objext.

Annotations/masks are yet to be implemented fully but the idea is that annotations will be standalone ojects that also have a view onto the metadata in the same way that scans currently do.

## Helper functions

In `pyescan.tools.dataset_utils` there are two useful dataset functions `summarise_dataset`, and `get_pe_export_summary`

`summarise_dataset` simply traverses a directory structure and finds all files matching a particular file structure, and summarises them in a single pandas dataframe.

For example
```
summarise_dataset([ROOT_DIRECTORY], structure="{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png"
```
Will get all files that look like `{pat}/{sdb}/{source_id}_{bscan_index:\d+}.png` (compared to the `[ROOT_DIRECTORY]`) and list them in a table with columns for `pat`, `sdb`, `source_id`, and `bscan_index` (as well as `file_path`). As you cna imagine this can be very useful for loading different types of input file structures.

The other function, `get_pe_export_summary`, is similar, but traverses through a Private/CrystalEye export and dives into the metadata, producing a dataframe with a full summary of all scans/images including images (there is some data still missing, in particular the image-level metadata, but the plan is to add this at some point). This is useful to make metadata summary CSVs which can be merged with other datasets (e.g. annotations/masks) as needed 


# Explanation + Development

Currently PyeScan is just about laying the groundwork in order to make things easier, but the idea is that we should be able to eventually port over functionality from existing code (and it should be much nicer / more streamlined to implement) Whenever you find a comon task you have to do constantly, consider adding it to the library!

The core object of PyeScan is the scan object which inherits from the `BaseScan` class (though some scan objects are also compositions of other BaseScan-derived instances, as well as being a `BaseScan` in their own right).

These scan classes provide a standard interface for various operations, along with various helper fucntionality.

A big part of how this is achieved is with `MetadataView` objects which (as the name suggests) provide a 'view' onto the raw metadata, translating from the original metadata structure (by prividing various acessor properties), while also handling managing which part of the metadata the particular scan refers to.

Loading of actual images is currently done 'lazily' so data is only loaded when actually needed. This is intended to be largely seamless for the end user, but obviously there may be times where the user wants to 'precache' data, so `preload` and `unload` functions are included for forcing loading/unloading.

Currently only loading images directly from disk is supported, but it should be possible to implement other ways of loading (e.g. directly from e2e file or whatever) fairly transparently.

Annotations/masks are still not implemented but the idea is that annotations will be standalone ojects that also have a view onto the metadata in the same way that scans currently do. There will be a system to properly identifyu which mdoel and which feature the annotation came from/refers to, and ways of automatically searching for annotations and linking them to certain scans.


## TODO

Main items:
- Annotations/masks
- Port over functionality from PEExportBase
- Proper documentation (+ Linting)