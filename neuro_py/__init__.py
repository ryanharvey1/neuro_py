from lazy_loader import attach as _attach

_proto_all_ = [
    "behavior",
    "ensemble",
    "io",
    "lfp",
    "plotting",
    "process",
    "session",
    "spikes",
    "stats",
    "tuning",
]

__getattr__, __dir__, __all__ = _attach(
    __name__, submodules=_proto_all_
)

del _attach
