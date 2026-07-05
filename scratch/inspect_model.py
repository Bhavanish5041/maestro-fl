import zipfile
import json
import torch
import io
import os

model_zip = "models/ppo_traffic_GS_cluster_10123822790_11303526453_11303526454_248766831_#2more_final.zip"
print("Model path exists:", os.path.exists(model_zip))

with zipfile.ZipFile(model_zip, 'r') as archive:
    # Print files inside zip
    print("Files inside zip:")
    for name in archive.namelist():
        print(" -", name)
        
    # Read data
    data_bytes = archive.read("data")
    data_str = data_bytes.decode('utf-8')
    data = json.loads(data_str)
    print("Data keys:")
    print("  observation_space:", data.get("observation_space"))
    print("  action_space:", data.get("action_space"))
    
    # Read parameters
    policy_bytes = archive.read("policy.pth")
    policy_dict = torch.load(io.BytesIO(policy_bytes), map_location="cpu")
    print("Policy state dict shapes:")
    for k, v in policy_dict.items():
        if "mlp_extractor" in k or "action" in k or "value" in k:
            print(f"  {k}: {v.shape}")
