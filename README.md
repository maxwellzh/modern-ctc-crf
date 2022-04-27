# modern-ctc-crf
Binding CTC-CRF loss C++/CUDA codes to modern PyTorch, ported from https://github.com/thu-spmi/CAT 

## RoadMap

Dependencies of current `ctc-crf` module:
```
ctc_crf
    -   gpu_ctc (warp-ctc)
    -   gpu_den
            -   openfst
    -   path_weight
            -   openfst
```

I plan to:
1. Deprecate the existing numerator function (CTC loss, borrowed from Baidu's warp-ctc) and use torch official CTCLoss instead, which means `gpu_ctc` is to be removed.
2. Use python package to load the denominator LM (in .fst format). This needs further digging into the `gpu_den` codes. If all goes well, building `openfst` from source will be removed. And I would provide a new python script to replace the current `path_weight`.
3. Update the existing CUDA codes to modern api. Functions like `cudaMalloc`, `cudaFree` and `cudaMemcpy` will all be replaced by higher level libtorch interfaces.
4. Add more help information and test cases to the repo.

Hopefully, this repo would contain only a few of files like
```
ctc_crf
    -   setup.py
    -   binding.cpp
    -   core.h
    -   core.cu
    -   ctc_crf
            -   __init__.py
            -   path_weight.py
    -   test
            ...
```


## Reference

1. PyTorch CTCLoss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss
2. OpenFST python: https://pypi.org/project/openfst-python/
3. An example of C++/CUDA PyTorch binding: https://github.com/maxwellzh/torch-gather