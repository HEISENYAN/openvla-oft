from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "Heisen0928/tfds_aloha_game_300"

files = list_repo_files(repo_id ,repo_type="dataset")
#files = [file for file in files if "openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_300_8--50000_chkpt" in file]
for file in files:
    print(file)
    hf_hub_download(
        repo_id=repo_id, 
        filename=file,
        repo_type="dataset",
        local_dir="/home/agilex/"
        )