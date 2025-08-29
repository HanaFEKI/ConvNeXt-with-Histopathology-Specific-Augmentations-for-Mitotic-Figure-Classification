import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from trainer import MitosisTrainer

# Load dataframe
csv_path = "/cluster/CBIO/data2/gbalezo/datasets/midog25_task2/dataframes/full_dataframe.csv"
df = pd.read_csv(csv_path)

train_root = Path("/cluster/CBIO/data2/gbalezo/datasets/midog25_task2/")
df["filepath"] = df["filepath"].apply(lambda x: str(train_root / x))

train_images = df["filepath"].tolist()
class_map = {"Atypical": 0, "Normal": 1}
train_labels = df["final_label"].map(class_map).tolist()

# Split train/val
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# Training config
trainer = MitosisTrainer(
    model_name='convnextv2_base.fcmae_ft_in22k_in1k',
    weights='IMAGENET1K_V1',
    num_epochs=20,
    batch_size=128,
    num_folds=5,
    lr=1e-4,
    experiment_dir='results_convnextv2'
)

# Train
val_accuracies = trainer.train(train_images=train_images, train_labels=train_labels)
