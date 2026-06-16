from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alexmkwizu/gaussian_training_datasets",
    repo_type="dataset",
    allow_patterns="tandt/truck/*",
    local_dir="./truck_dataset"
)