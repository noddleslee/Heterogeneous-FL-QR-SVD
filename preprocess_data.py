"""
Pre-processing script to filter and convert PathVQA dataset once.
"""

import os, warnings
from datasets import load_dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message="Truncated File Read")

_SAVE_DIR = "/mimer/NOBACKUP/groups/bloom/shenghui/hfl/processed_pathvqa"

def _is_valid_image(sample):
    try:
        sample["image"].convert("RGB").load()
        return True
    except Exception:
        return False

def _to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text",  "text": sample["question"]},
                {"type": "image", "image": sample["image"].convert("RGB")},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["answer"]},
            ]},
        ]
    }

def preprocess():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    
    for split in ["train", "validation"]:
        print(f"\nProcessing {split} split …")
        ds = load_dataset("flaviagiammarino/path-vqa", split=split)
        
        print(f"  Filtering {split} …")
        ds = ds.filter(_is_valid_image, num_proc=1)
        
        print(f"  Converting {split} …")
        ds = ds.map(_to_conversation, num_proc=1, remove_columns=ds.column_names)
        
        save_path = os.path.join(_SAVE_DIR, split)
        print(f"  Saving to {save_path} …")
        ds.save_to_disk(save_path)

    print("\nPre-processing complete.")

if __name__ == "__main__":
    preprocess()
