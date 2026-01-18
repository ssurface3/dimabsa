import numpy as np

MIN_VAL = 1.0
MAX_VAL = 9.0
STEP = 0.25
NUM_BINS = int((MAX_VAL - MIN_VAL) / STEP) 

def float_to_bin(val):
    """
    Converts a score (e.g. 6.12) to a Bin ID (e.g. 20).
    If the score is outside [1.0, 9.0], it is clipped to the nearest valid bin.
    1.0 maps to bin 0, 9.0 maps to bin 31.
    1.0-1.25 -> bin 0
    1.25-1.5 -> bin 1
    """
    # Normalize to 0-offset
    val = float(val) - MIN_VAL
    
    # Divide by step size
    bin_idx = int(val / STEP)
    
    # Safety Clip (0 to 31)
    return max(0, min(NUM_BINS - 1, bin_idx))

def bin_to_float(bin_idx):
    """
    Converts a Bin ID (e.g. 20) back to a Score (e.g. 6.125).
    We return the CENTER of the bin.
    """
    val = (bin_idx * STEP) + MIN_VAL
    
    center_val = val + (STEP / 2.0)
    
    return center_val

