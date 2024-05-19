import torch
from torch import nn
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
from mlflow import log_metric, log_param, log_artifacts
from torcheval.metrics import BinaryF1Score, BinaryAccuracy
import mlflow
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class CatsDogsDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label2int = {"Cat":0, "Dog":1}
        self.image_cache = {}
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']

        if image_path not in self.image_cache:
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = np.array(image)
            image = torch.from_numpy(image)
            image = image.float() / 255
            image = image.permute(2, 0, 1)
            self.image_cache[image_path] = image
        else:
            image = self.image_cache[image_path]

        label = self.df.iloc[idx]['image_class']
        # print(image.shape)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor([self.label2int[label]], dtype=torch.float32)

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    mlflow.enable_system_metrics_logging()

    train_df, valid_df = train_test_split(
        df,
        test_size=0.1,
        random_state=seed,
        stratify=df["image_class"]
    )
    final_size = 224
    train_transform = transforms.Compose([
        transforms.ToPILImage(),                # Convert the image to a PIL Image
        transforms.Resize((final_size, final_size)), # Resize the image to final_size x final_size
        # transforms.RandomResizedCrop(final_size), # Crop the image to a random size and aspect ratio
        transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
        transforms.ColorJitter(0.2, 0.2, 0.2), # Randomly adjust brightness, contrast, saturation, and hue
        transforms.RandomRotation(20),         # Randomly rotate the image by up to 20 degrees
        transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),                # Convert the image to a PIL Image
        transforms.Resize((final_size, final_size)), # Resize the image to final_size x final_size
        transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
    ])

    train_ds = CatsDogsDataset(train_df, transform=train_transform)
    valid_ds = CatsDogsDataset(valid_df, transform=valid_transform)
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda")

    num_epochs = 20

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    f1_score = BinaryF1Score()
    accuracy_score = BinaryAccuracy()

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_f1": [],
        "valid_f1": []
    }

    best_loss = float("inf")
    best_f1 = 0
    best_accuracy = 0
    mlflow.set_experiment("PyTorch_cats_dogs")
    with mlflow.start_run():
        log_param("num_epochs", num_epochs)
        log_param("batch_size", batch_size)
        log_param("seed", seed)
        log_param("final_size", final_size)
        log_param("model", "resnet50")
        log_param("optimizer", "Adam")
        log_param("criterion", "BCELoss")

        mlflow.log_artifact("data/data.csv")
        mlflow.log_artifact(__file__)

        for epoch_idx in range(num_epochs):
            train_loss = 0
            valid_loss = 0
            train_accuracy = 0
            valid_accuracy = 0
            train_f1 = 0
            valid_f1 = 0

            model.train()
            for x, y in tqdm(train_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(torch.sigmoid(output), y)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                f1_score.update(torch.sigmoid(output).squeeze(), y.squeeze())
                accuracy_score.update(torch.sigmoid(output).squeeze(), y.squeeze())

            history["train_loss"].append(train_loss / len(train_loader))
            history["train_accuracy"].append(accuracy_score.compute())
            history["train_f1"].append(f1_score.compute())

            accuracy_score.reset()
            f1_score.reset()

            mlflow.log_metric("train_loss", history["train_loss"][-1], step=epoch_idx)
            mlflow.log_metric("train_accuracy", history["train_accuracy"][-1], step=epoch_idx)
            mlflow.log_metric("train_f1", history["train_f1"][-1], step=epoch_idx)
            model.eval()
            with torch.no_grad():
                for x, y in tqdm(valid_loader):
                    x, y = x.to(device), y.to(device)

                    output = model(x)
                    loss = criterion(torch.sigmoid(output), y)
                    valid_loss += loss.item()

                    f1_score.update(torch.sigmoid(output).squeeze(), y.squeeze())
                    accuracy_score.update(torch.sigmoid(output).squeeze(), y.squeeze())

            history["valid_loss"].append(valid_loss / len(valid_loader))
            history["valid_accuracy"].append(accuracy_score.compute())
            history["valid_f1"].append(f1_score.compute())

            accuracy_score.reset()
            f1_score.reset()

            if history["valid_loss"][-1] < best_loss:
                best_loss = history["valid_loss"][-1]
                print(f"Found better loss: {best_loss}")
                torch.save(model.state_dict(), "best_loss.pth")
                mlflow.pytorch.log_model(model, "best_loss")

            if history["valid_f1"][-1] > best_f1:
                best_f1 = history["valid_f1"][-1]
                print(f"Found better f1: {best_f1}")
                torch.save(model.state_dict(), "best_f1.pth")
                mlflow.pytorch.log_model(model, "best_f1")

            if history["valid_accuracy"][-1] > best_accuracy:
                best_accuracy = history["valid_accuracy"][-1]
                print(f"Found better accuracy: {best_accuracy}")
                torch.save(model.state_dict(), "best_accuracy.pth")
                mlflow.pytorch.log_model(model, "best_accuracy")
            
            mlflow.log_metric("valid_loss", history["valid_loss"][-1], step=epoch_idx)
            mlflow.log_metric("valid_accuracy", history["valid_accuracy"][-1], step=epoch_idx)
            mlflow.log_metric("valid_f1", history["valid_f1"][-1], step=epoch_idx)

            print(
                f"Epoch {epoch_idx + 1}/{num_epochs} "
                f"Loss: {history['train_loss'][-1]:.4f}/{history['valid_loss'][-1]:.4f} "
                f"Accuracy: {history['train_accuracy'][-1]:.4f}/{history['valid_accuracy'][-1]:.4f} "
                f"F1: {history['train_f1'][-1]:.4f}/{history['valid_f1'][-1]:.4f}"
            )

        pd.DataFrame(history).to_csv("history.csv", index=False)
        mlflow.log_artifact("history.csv")

