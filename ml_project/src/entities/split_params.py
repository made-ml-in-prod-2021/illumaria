from dataclasses import dataclass, field


@dataclass()
class SplitParams:
    random_state: int = field(default=4)
    val_size: float = field(default=0.2)
