from typing import Any, Dict, Callable, Optional

class PandasRowWrapperHelper:
    def __init__(self,
                 row: Any,
                 column_map: Optional[Dict[str, str]] = None,
                 special_funcs: Optional[Dict[str, Callable[..., Any]]] = None):
        self._row = row
        self._column_map = column_map or {}
        self._spec_funcs = special_funcs or {}
        self._cache = {}
        
    def __getattr__(self, name: str) -> Any:
        """Get column value by attribute name"""
        # Check for special mappings first
        if name in self._spec_funcs:
            func = self._spec_funcs[name]
            return func(self)
        elif name in self._column_map:
            mapped_name = self._column_map[name]
            return self._row.loc[mapped_name]
            
        # Try direct column access
        try:
            return self._row.loc[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
            
    def __getitem__(self, key: str) -> Any:
        """Get column value by string key"""
        return self.__getattr__(key)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking column existence"""
        return key in self._column_map or key in self._spec_funcs or key in self._row.index
    
