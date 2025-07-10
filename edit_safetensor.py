from safetensors.torch import load_file, save_file
import torch

# Load original safetensors
data = load_file("results/ambiguous_qe_inf/test_save/best_model/adapter_model.safetensors")
data_2 = load_file("results/nq_inf/toy_contrastive/checkpoint_70000/adapter_model.safetensors")

# Create a new dictionary with modified keys
new_data = {}
# for k, v in data.items():
#     print(k, v.shape)
for key, tensor in data.items():
    # Example: rename key by replacing "old" with "new"
    # new_key = key.replace("base_causallm.", "")
    new_data[key] = tensor
    
for key, tensor in data_2.items():
    print(tensor[0])
    print(new_data[key][0])
    break

# # Save to a new file
# save_file(new_data, "results/ambiguous_qe_inf/test_save_safe/best_model/adapter_model_modified.safetensors")