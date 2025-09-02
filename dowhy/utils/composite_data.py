from collections.abc import Hashable, Iterator
from itertools import chain
import pandas as pd
from pandas.core.indexing import _AtIndexer, _iAtIndexer, _LocIndexer, _iLocIndexer


class CompositeDataFrame(pd.DataFrame):
    def __init__(self, dataframes: list[pd.DataFrame]) -> None:
        self.data = dataframes
        super().__init__()


    def __contains__(self, key):
        """True if the key is in the info axis"""
        return any(key in df._info_axis for df in self.dataframes)

    def __iter__(self) -> Iterator[Hashable]:
        iter = chain(*self.dataframes)
        return iter

    def __setitem__(self, key, value) -> None:
        raise NotImplementedError()

    @property
    def iloc(self) -> _iLocIndexer:
        raise NotImplementedError()
   
        return _iLocIndexer("iloc", self)

    @property
    def loc(self) -> _LocIndexer:
        raise NotImplementedError()

        return _LocIndexer("loc", self)

    @property
    def at(self) -> _AtIndexer:
        raise NotImplementedError()

        return _AtIndexer("at", self)
 
    @property
    def iat(self) -> _iAtIndexer:
        raise NotImplementedError()

        return _iAtIndexer("iat", self)
