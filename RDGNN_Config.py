from dataclasses import dataclass, field
from typing import List

@dataclass
class RDGNN_Config:

    #Example field for config:
    # color: str = "default value"

    
    max_objects: int = 8
    environment_object_names: List = field(default_factory=lambda : ['table'])
    one_hot_encoding_dim: int = 128
    node_emb_size: int = 128
    edge_emb_size: int = 128
    relation_output_size: int = 10
