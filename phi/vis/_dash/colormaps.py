# Scivis Colormaps
# https://sciviscolor.org/
import warnings

import numpy as np

from phi.math.backend import PHI_LOGGER

ORANGE_WHITE_BLUE = [
    [0.      ,  22.00000005,   1.00000035,  76.0000011 ],
    [0.030334,  28.999875  ,   5.999997  , 114.9999    ],
    [0.055527,  26.99991   ,  13.000002  , 130.00002   ],
    [0.073008,  10.0000035 ,  10.0000035 , 142.99992   ],
    [0.089974,   7.9999875 ,  24.999996  , 153.        ],
    [0.106427,  11.0000115 ,  42.00003   , 163.00008   ],
    [0.130077,  14.00001   ,  61.999935  , 172.999905  ],
    [0.16144 ,  14.00001   ,  80.999985  , 181.00002   ],
    [0.2     ,  13.000002  , 100.99989   , 188.99988   ],
    [0.225   ,  10.0000035 , 119.000085  , 195.999885  ],
    [0.25    ,   7.9999875 , 137.000025  , 200.999925  ],
    [0.276093,   7.9999875 , 156.99993   , 207.000075  ],
    [0.302828,   5.999997  , 181.00002   , 212.000115  ],
    [0.329563,  13.000002  , 204.        , 216.9999    ],
    [0.351671,  17.999991  , 218.00001   , 221.99994   ],
    [0.372237,  66.999975  , 230.000055  , 219.999975  ],
    [0.390231, 107.999895  , 239.99988   , 223.00005   ],
    [0.417995, 145.999995  , 246.00003   , 212.99997   ],
    [0.436504, 168.00012   , 249.99996   , 214.999935  ],
    [0.456041, 195.00003   , 249.99996   , 221.000085  ],
    [0.468895, 211.000005  , 249.99996   , 226.000125  ],
    [0.482262, 227.0000004 , 251.99999895, 236.000001  ],
    [0.492545, 232.999875  , 251.999925  , 239.000025  ],
    [0.5     , 244.0966743 , 253.51315365, 243.539586  ],  # Inserted
    [0.501285, 255.        , 255.        , 248.0000001 ],
    [0.510026, 251.99999895, 251.99999895, 230.99999925],
    [0.526478, 252.9999993 , 248.0000001 , 205.00000035],
    [0.539846, 253.000035  , 246.00003   , 181.999875  ],
    [0.554756, 251.999925  , 244.000065  , 163.999935  ],
    [0.576864, 249.99996   , 233.999985  , 130.00002   ],
    [0.599486, 246.999885  , 223.00005   , 103.999965  ],
    [0.620051, 242.0001    , 209.999895  ,  82.000095  ],
    [0.636504, 237.00006   , 198.000105  ,  70.999905  ],
    [0.660668, 232.00002   , 182.999985  ,  59.99997   ],
    [0.682262, 226.99998   , 168.00012   ,  49.99989   ],
    [0.7     , 223.999905  , 158.00004   ,  42.999885  ],
    [0.725   , 221.99994   , 140.0001    ,  40.000065  ],
    [0.75    , 216.9999    , 121.00005   ,  36.99999   ],
    [0.775   , 212.000115  , 105.000075  ,  33.999915  ],
    [0.8     , 207.000075  ,  87.99999   ,  28.999875  ],
    [0.825   , 200.999925  ,  68.000085  ,  23.999988  ],
    [0.85    , 188.99988   ,  47.00007   ,  18.999999  ],
    [0.875   , 175.99998   ,  31.99995   ,  16.0000005 ],
    [0.9     , 158.00004   ,  16.0000005 ,  11.0000115 ],
    [0.923393, 140.0001    ,   7.000005  ,  17.999991  ],
    [0.943959, 119.99994   ,   4.0000065 ,  23.0000055 ],
    [0.967095, 102.        ,   1.00000035,  26.000055  ],
    [1.      ,  47.99999895,   0.        ,  18.0000012 ]]

BLUE_WHITE_RED = [[0, 0, 0, 220],
                  [0.5, 220, 220, 220],
                  [1,   220,   0,   0]]


VIRIDIS_EXTENDED = [
    [0.0, 255, 200,100],
    [.13, 255, 153, 51],
    [.25, 230,  5,  40],
    [.38, 150,  3,  62],
    [0.5, 68,   1,  84],
    [.55, 72,  33, 115],
    [.59, 67,  62, 133],
    [.64, 56,  88, 140],
    [.68, 45, 112, 142],
    [.73, 37, 133, 142],
    [.77, 30, 155, 138],
    [.82, 42, 176, 127],
    [.86, 82, 197, 105],
    [.90, 34,  213, 73],
    [.95, 194, 223, 35],
    [1.0, 253, 231, 37]
]


COLORMAPS = {
    None: VIRIDIS_EXTENDED,
    'OrWhBl': ORANGE_WHITE_BLUE,
    'viridisx': VIRIDIS_EXTENDED,
}


# --- Load available Matplotlib color maps ---
try:
    from matplotlib.pyplot import colormaps
    from matplotlib.colors import ListedColormap
    from matplotlib import colormaps as cms

    for name in colormaps():
        colormap = cms[name]
        if isinstance(colormap, ListedColormap):
            pos = np.expand_dims(np.linspace(0, 1, len(colormap.colors)), axis=-1)
            cols = np.array(colormap.colors) * 255
            COLORMAPS[name] = np.concatenate([pos, cols], axis=-1)
except ImportError:
    warnings.warn('matplotlib is not installed. Corresponding colormaps are not available.', ImportWarning)
