# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from src.data import classification_dataloader, pretrain_dataloader, custom_datasets, channels_strategies
# 这个是data的init文件，其中会 import classification_dataloader, pretrain_dataloader, custom_datasets,
# channels_strategies这些文件，但这些文件中如果有报错，比如它们会import一个在环境中没有的包，那么整个import也会报错，即
# 这个init会报错，连带着最上层how_to_use那边import chadavit也会报错
__all__ = [
    "classification_dataloader",
    "pretrain_dataloader",
    "custom_datasets",
    "channels_strategies",
]


try:
    from src.data import dali_dataloader  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali_dataloader")
