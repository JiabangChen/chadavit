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

from .vit import vit_ultra_tiny, vit_tiny, vit_small, vit_base, vit_large, vit_channels
# 这个是backbone的init.py文件，其中第一步是先调用backbone的vit包，因此会先跑vit的init文件，然后import
# vit包中的vit_ultra_tiny, vit_tiny, vit_small, vit_base, vit_large, vit_channels这些函数，但这些
# 函数已经在vit的init文件中定义过了
__all__ = [
    "vit_ultra_tiny",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_channels",
]
