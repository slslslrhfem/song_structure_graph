import numpy as np
from matplotlib import pyplot as plt

class pattern:
    """
    A class for "pattern node", which contains information of instruments, start time(index), and notes(with pianoroll) 
    """
    def __init__(
        self,
        inst: str = None,
        pattern_order : int = None,
        start_timing : int = None,
        key : int = None,
        genre : str = None,
        CONLON : np.ndarray = None):
        self.inst = inst if inst is not None else 'PIANO'
        self.pattern_order = pattern_order if pattern_order is not None else 0
        self.start_timing = start_timing if start_timing is not None else 0 #actually 0 is not meanless value, but set pattern_order/start_timing 0 if there is minor issue. For preventing heavy collision..
        self.key = key if key is not None else 35 # 0 is for C major, 35 is meaningless integer. we excluded music data without key, so there is no 35 for any key data if there is no minor issue...
        self.genre = genre if genre is not None else 'None'
        if CONLON is not None:
            self.CONLON = CONLON
        else:
            raise Exception('pattern should contain pianoroll( or CONLON ) information.')

    def __repr__(self) -> str:
        to_join = [
            f"inst={repr(self.inst)}",
            f"pattern_order={repr(self.pattern_order)}",
            f"start_timing={repr(self.start_timing)}",
            f"key={repr(self.key)}",
            f"genre={repr(self.genre)}",
            f"CONLON=array(shape={self.CONLON.shape}",
            f"dtype={self.CONLON.dtype})",
        ]
        return f"Track({', '.join(to_join)})"
