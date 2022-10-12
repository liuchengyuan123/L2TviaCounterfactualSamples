from typing import Dict, List
class Edition(object):
    
    def __init__(self) -> None:
        pass

    def load(self, data: List[Dict]) -> List[Dict]:
        return data
    
    def edit(self, sample: Dict) -> Dict :
        return sample
    
    def group_edit(self, samples: List[Dict]) -> List[Dict]:
        w = []
        for sample in samples:
            w.append(self.edit(sample))
        return w
