"""Upload the final checkpoint to the Hugging Face Hub.

1. Create a free account at huggingface.co and a token with *write* access
   (Settings > Access Tokens).
2. In Colab or the Codespace:
       export HF_TOKEN=hf_xxx          # in Colab: import os; os.environ['HF_TOKEN']='hf_xxx'
       python src/upload_checkpoint.py --ckpt /content/drive/MyDrive/ckpt_main/epoch14.pth \
              --repo <your-hf-username>/leaf-maskrcnn
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="username/repo-name")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=args.ckpt, path_in_repo="maskrcnn_leaf.pth",
                    repo_id=args.repo, repo_type="model")
    print(f"Uploaded. Download link: https://huggingface.co/{args.repo}/resolve/main/maskrcnn_leaf.pth")


if __name__ == "__main__":
    main()
