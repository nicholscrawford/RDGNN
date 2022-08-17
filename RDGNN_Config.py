from dataclasses import dataclass, field
from typing import List

@dataclass
class RDGNN_Config:

    #Example field for config:
    # color: str = "default value"

    
    one_hot_encoding_dim: int = 4
    max_objects: int = 8
    environment_object_names: List = field(default_factory=lambda : ['table'])
    node_emb_size: int = 128
    edge_emb_size: int = 128
