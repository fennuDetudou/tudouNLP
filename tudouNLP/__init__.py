# -*- coding: utf-8 -*-

"""
 @Time    : 2019/3/12
 @Author  : tudou (upczyxl@163.com)
 @File    : __init__.py.py
"""

import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
sys.path.append(os.path.join(current_dir,'models'))
sys.path.append(os.path.join(current_dir,'tools'))

from .models import train
from .models import predict
from .tools import utils
