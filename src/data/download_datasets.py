import os
from datasets import load_dataset


def download_goemotions():
    print("Downloading GoEmotions...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    os.makedirs("data/raw/goemotions", exist_ok=True)
    ds.save_to_disk("data/raw/goemotions")
    print(f"GoEmotions — Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")
    return ds


def download_empathetic_dialogues():
    print("Downloading Empathetic Dialogues...")
    ed = load_dataset("facebook/empathetic_dialogues")
    os.makedirs("data/raw/empathetic_dialogues", exist_ok=True)
    ed.save_to_disk("data/raw/empathetic_dialogues")
    print(f"Empathetic Dialogues — Train turns: {len(ed['train'])}")
    return ed


if __name__ == "__main__":
    download_goemotions()
    download_empathetic_dialogues()
    print("\nAll downloads complete.")
