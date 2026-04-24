# boxcox, cube_root, scaling


import numpy as np
from scipy.stats import boxcox

def transform_features(df):
    df = df.copy()
    df["cap-diameter"], _ = boxcox(df["cap-diameter"] + 1e-6)
    df["stem-height"] = np.cbrt(df["stem-height"])
    return df
