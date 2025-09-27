from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "fangqi/openvla-oft"

files = list_repo_files(repo_id )#,repo_type="dataset")
files = [file for file in files if "09/23/aloha_wm_128_grpo_b64/global_step_15" in file]
print(files)
for file in files:
    print(file)
    hf_hub_download(
        repo_id=repo_id, 
        filename=file,
        #repo_type="dataset",
        local_dir="/home/agilex/checkpoints"
        )