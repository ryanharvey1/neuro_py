import sys
from gc import get_referents

import neuro_py


def getsize(obj):
    """sum size of object & members."""
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            # print(obj)
            if id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

def test_lazy_loader():
    # default module size
    module_size = getsize(neuro_py)

    # load submodules
    neuro_py.io.loadLFP

    # get new module size
    new_module_size = getsize(neuro_py)

    # check that the new module size is larger than the default
    assert new_module_size > module_size
