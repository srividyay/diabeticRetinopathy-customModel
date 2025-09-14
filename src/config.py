from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class Config:
    data: Dict[str, Any]
    paths: Dict[str, Any]
    training: Dict[str, Any]
    gpu: Dict[str, Any]
    augment: Dict[str, Any]
    streamlit: Dict[str, Any]
    severity: Dict[str, Any]
    project_name: str
    random_seed: int

def load_config(path: str) -> 'Config':
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Config(
        project_name=cfg.get('project_name','image_pipeline'),
        random_seed=cfg.get('random_seed', 42),
        data=cfg['data'],
        paths=cfg['paths'],
        training=cfg['training'],
        gpu=cfg.get('gpu', {}),
        augment=cfg.get('augment', {}),
        streamlit=cfg.get('streamlit', {}),
        severity=cfg.get('severity', {}),
    )
