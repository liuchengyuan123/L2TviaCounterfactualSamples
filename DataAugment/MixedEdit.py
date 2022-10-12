"""
Edit data columns into other string .
No edit topic
"""

from collections import defaultdict
from copy import copy
import os
from typing import Dict, Iterable, List, Optional
import numpy as np
import string
import json

from tqdm import tqdm

from DataAugment.EditAbstract import Edition
from DataAugment.randEdit import RDEdition
# from DataAugment.normEdit import normEdition
from DataAugment.dtypeEdit import dtypeEdition

class MixedEdition(Edition):
    def __init__(self, prob=1) -> None:
        super().__init__()
        self.prob = prob
        self.rand_editor = RDEdition(prob)
        self.norm_editor = dtypeEdition(prob)
        self.count = 1

    def load(self, data: List[Dict]) -> List[Dict]:
        sample_new = self.norm_editor.load(data)
        return sample_new

    def edit(self, *args, **kwargs) -> Optional[Dict]:
        if self.count == 0:
            sample = self.rand_editor.edit(*args, **kwargs)
        else:
            sample = self.norm_editor.edit(*args, **kwargs)
        self.count = 1 - self.count
        return sample

    def check_replacable(self, s:str) -> bool:
        return s.isalnum() and not s.isnumeric()

