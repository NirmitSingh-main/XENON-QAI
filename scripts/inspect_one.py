from pathlib import Path
import h5py
import numpy as np

path = Path("data/extracted/ER/ER_all.hdf5")

with h5py.File(path, "r") as f:
    for name in ["HITS_bot", "HITS_bot_img", "HITS_top", "HITS_top_img", "peakS1_onlyhits", "peakS1_raw", "peakS2_onlyhits", "peakS2_raw"]:
        data = f[name]["ev_1"][:]

        print("\n" + name)
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("Min:", np.min(data))
        print("Max:", np.max(data))
        print("Mean:", np.mean(data))
        print("Non-zero:", np.count_nonzero(data))

    bot_img = f["HITS_bot_img"]["ev_1"][:]
    top_img = f["HITS_top_img"]["ev_1"][:]

    print("\nIMAGE CHECK")
    print("Bottom image:", bot_img.shape)

    if bot_img.size == 128 * 128:
        print("Bottom image can be reshaped to:", bot_img.reshape(128, 128).shape)

    print("Top image:", top_img.shape)

    if top_img.size == 128 * 128:
        print("Top image can be reshaped to:", top_img.reshape(128, 128).shape)