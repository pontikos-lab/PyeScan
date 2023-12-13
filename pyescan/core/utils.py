import numpy as np

class ArrayView():
    # Helper wrapper class for lists of objects which can be converted to data
    def __init__(self, items):
        self._items_list = items
        
    def _items():
        # Overwrite to return item class
        return self._items_list
      
    def __len__(self):
        return len(self._items())
    
    def __getitem__(self, index):
        if index is Ellipsis:
            return self.data
        elif isinstance(index, tuple):
            # In this case we want to deliver the raw data for convenience
            # Slice array along first index, then use numpy slicing for rest
            if index[0] is Ellipsis:
                # Special case as ellipsis can signify multiple indices
                data = self.data
                if len(index[1:]) == len(data.shape):
                    return data[(slice(None),) + index[1:]]
                else:
                    return data[index]
            elif isinstance(index[0], slice):
                return self[index[0]].data[(slice(None),) + index[1:]]
            else:
                return self[index[0]].data[index[1:]]     
        elif isinstance(index, slice):
            return self.__class__(self._items()[index])
        else:
            return self._items()[index]
    
    def __iter__(self):
        return iter(self._items())
    
    def __str__(self):
        return f"{self.__class__.__name__}({str(self._items())})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._items())})"
    
    @property
    def data(self):
        return np.array([ np.array(item) for item in self._items() ])
    
    def __array__(self):
        return self.data