import os
from huggingface_hub import login, snapshot_download

login(token=os.environ["HF_TOKEN"])  # set HF_TOKEN env var before running

snapshot_download(
    repo_id="l8cv/GenBuster-200K-mini",
    repo_type="dataset",
    local_dir="/workspace/data/genbuster-mini"
)