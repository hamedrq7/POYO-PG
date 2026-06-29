import torch_brain
print(torch_brain.__version__)

import numpy as np 
import re
def regex_for_suffix(s2: str) -> str:
    return r".*" + re.escape(s2) + r"$"

def _generate_unit_mask(pattern, units) -> np.ndarray:
    unit_mask = np.array([bool(pattern.search(uid)) for uid in units])
    unit_mask = ~unit_mask
    return unit_mask

us = [f'asd/_unit_{i}' for i in range(100)]
print(us)
p = regex_for_suffix("unit_2")
print((p))
print(_generate_unit_mask(p, us))

"""
185.49.84.2
102.37.12.10
102.37.16.154
102.37.16.134
102.37.16.199
194.225.62.66
178.252.184.139
8.8.8.8        
1.1.1.1        
9.9.9.9        
208.67.222.222 
94.140.14.14
185.228.168.9
8.8.4.4
8.26.56.26


185.49.84.2:53	
194.225.62.66:53
178.252.184.139:53	
"""