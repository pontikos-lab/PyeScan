import numpy as np
import pandas as pd
import tqdm
from typing import Any, Dict, Callable, List, Optional, Union

from .processor import MetricProcessor
from .registry import PYESCAN_GLOBAL_METRICS

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
        except KeyError as e:
            #raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
            raise e
            
    def __getitem__(self, key: str) -> Any:
        """Get column value by string key"""
        return self.__getattr__(key)
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking column existence"""
        return key in self._column_map or key in self._spec_funcs or key in self._row.index
    
def run_on_dataframe(df,
                     stat_name:Union[str,List[str]],
                     col_mapping:Dict[str,str] = {},
                     auto_merge: bool = False,
                     named_only: bool = False,
                     metric_list=None):
    
    metrics = metric_list or PYESCAN_GLOBAL_METRICS.metrics
    processor = MetricProcessor(metrics)
    
    if isinstance(stat_name, str): #Make list for unified processing
        stat_name = [stat_name] 
        
    metrics = [ processor.get_metric_by_stat(s) for s in stat_name ]
    
    results = []
    for idx, row in tqdm.tqdm(df.iterrows()):
        wrapped_row = PandasRowWrapperHelper(row, col_mapping)
        
        result = dict()
        cache = dict()
        for metric, params in metrics:
            cache = processor._process_metric(wrapped_row, metric, params, cache=cache)
            result.update(cache['computed_stats'])

        if named_only:
            filtered_result = {k: v for k, v in result.items() if k in stat_name}
        else:  
            filtered_result = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        results.append(filtered_result)
    
    df_stats = pd.DataFrame(results)
    
    if auto_merge:
        return pd.concat([df, df_stats], axis=1)
    else:
        return df_stats