from dataclasses import dataclass
from typing import List

@dataclass
class RDGNN_Config:

    #Example field for config:
    # color: str = "default value"

    
    one_hot_encoding_dim: int = 4
    max_objects: int = 8
    environment_object_names: List = ['table']
