import os
import pandas as pd
from sklearn.model_selection import train_test_split

HYPOTHESIS = "This person is expressing suicidal ideation or intent to self-harm."


def build_nli_pairs(
    input_path="data/raw/suicide_detection/Suicide_Detection.csv",
    output_dir="data/processed"
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text", "class"])

    # Map to NLI labels: entailment=0 (crisis), contradiction=1 (non-crisis)
    df["nli_label"] = df["class"].map({"suicide": 0, "non-suicide": 1})
    df = df.dropna(subset=["nli_label"])
    df["nli_label"] = df["nli_label"].astype(int)
    df["hypothesis"] = HYPOTHESIS

    # 80/10/10 stratified split
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df["nli_label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["nli_label"], random_state=42
    )

    train.to_csv(f"{output_dir}/nli_train.csv", index=False)
    val.to_csv(f"{output_dir}/nli_val.csv", index=False)
    test.to_csv(f"{output_dir}/nli_test.csv", index=False)

    print(f"NLI pairs — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Label distribution:\n{train['nli_label'].value_counts()}")


if __name__ == "__main__":
    build_nli_pairs()
