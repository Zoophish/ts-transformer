from typing import Type

import torch
import torch.nn as nn

from .stateful import StatefulModule


class DiskModule(nn.Module):
    """
    A proxy module that is like a standard nn.Module, but loads its internal
    weights from disk on demand. Provides load/unload so it can be managed by
    a cache.
    """
    def __init__(self, base_cls, hyperparams, path):
        super().__init__()
        self.base_cls = base_cls
        self.hyperparams = hyperparams
        self.path = str(path)
        
        self._model = None

        # tracks what device this gets loaded onto by .to()
        self.register_buffer('_device_tracker', torch.empty(0))

    @property
    def device(self):
        return self._device_tracker.device

    def load(self):
        if self._model is None:
            # instantiate
            self._model = self.base_cls(**self.hyperparams)
            
            # load weights (to cpu first to save VRAM during transfer)
            # assuming the path is a state_dict
            state = torch.load(self.path, map_location='cpu')
            self._model.load_state_dict(state)
        
        # ensure it's on the right device (even if already loaded)
        if self._model.device != self.device:
            self._model.to(self.device)
        
        return self

    def unload(self):
        if self._model is not None:
            # move to cpu first to detach from gpu context
            self._model.to('cpu') 
            del self._model
            self._model = None

            torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        if self._model is None:
            raise RuntimeError("DiskModule accessed before .load() was called!")
        return self._model(*args, **kwargs)

    def __getstate__(self):
        """
        When torch.save(ensemble) is called, this ensures we save the config/path,
        but not the heavy loaded model.
        """
        state = self.__dict__.copy()
        state['_model'] = None  # wipe the heavy model from the saved state
        return state
    

class DiskModuleCache(nn.Module):
    """
    Like a ModuleList, but keeps the most used module weights loaded. Evicts
    (unloads) the least recently used module if the cache size is exceeded.
    """
    def __init__(self, cache_size: int):
        super().__init__()
        self.disk_modules = nn.ModuleList()
        self.cache_size = cache_size
        self.cache_order = []

    def clear(self):
        for module in self.disk_modules:
            module.unload()
        self.cache_order = []

    def append(self, disk_module : DiskModule):
        self.disk_modules.append(disk_module)

    def __getitem__(self, idx: int):
        if idx in self.cache_order:
            # move to front
            self.cache_order.remove(idx)
            self.cache_order.insert(0, idx)
        else:
            # load and add to front
            self.disk_modules[idx].load()
            self.cache_order.insert(0, idx)
            # evict LRU if over capacity
            if len(self.cache_order) > self.cache_size:
                evict_idx = self.cache_order.pop()
                self.disk_modules[evict_idx].unload()

        return self.disk_modules[idx]
    
    def __setitem__(self, idx, disk_module):
        self.disk_modules[idx] = disk_module
        if idx in self.cache_order:
            self.cache_order.remove(idx)


class Ensemble(StatefulModule):
    """
    Ensemble of base models using stateful paradigm.

    The sub-model is sampled using the generator, so if the model generator is in the same state,
    it will yield the same result.

    Assumes n distinct model state files exist for the underlying base module class which are stored
    at the path prefix, postfixed by and underscore and the index (e.g. 'mymodel_1').
    """

    def __init__(
            self,
            base : Type[nn.Module],
            hyperparams : dict,
            n : int = 1,
            cache_size : int = 1,
            path_prefix : str = None
        ):
        super().__init__()
        self.base = base
        self.n = n
        self.generator = None
        self.net = DiskModuleCache(cache_size)
        self.hyperparams = hyperparams
        self.cache_size = cache_size
        self.path_prefix = path_prefix

        self.build()

    def build(self):
        for i in range(self.n):
            self.net.append(
                DiskModule(
                    base_cls=self.base,
                    hyperparams=self.hyperparams,
                    path=f"{self.path_prefix}_{i}"
                )
            )

    def forward(self, *args, **kwargs):
        if self.generator is None:
            raise RuntimeError("Ensemble generator must not be None.")
        
        model_idx = torch.randint(0, len(self.net), [1], generator=self.generator).item()
        return self.net[model_idx](*args, **kwargs)
    
    def __len__(self):
        return len(self.net)
    
    def __getitem__(self, idx : int):
        return self.net[idx]
    