from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "fangqi/openvla-oft-dpo"

files = list_repo_files(repo_id )#,repo_type="dataset")
#files = [file for file in files if "09/04/aloha_game_100_fix/openvla-7b+aloha_game+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--aloha_game_100_8_fix--100000_chkpt" in file]
for file in files:
    print(file)
    hf_hub_download(
        repo_id=repo_id, 
        filename=file,
        #repo_type="dataset",
        local_dir="/home/agilex/checkpoints/aloha-game-dpo"
        )