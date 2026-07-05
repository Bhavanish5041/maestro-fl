import pickle
import os

pkl_path = "global_params.pkl"
print("global_params.pkl exists:", os.path.exists(pkl_path))
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
    print("global_params.pkl contents length:", len(params))
    for i, p in enumerate(params):
        try:
            print(f"Param {i}: shape={p.shape}")
        except AttributeError:
            print(f"Param {i}: type={type(p)}")
