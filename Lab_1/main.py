import os
import torch
from utils.utils import download_data_and_clear_cache, plot_metrics
from dataloader.MNIST import MNISTDataset
from torch.utils.data import DataLoader
from models.one_layer_MLP import OneLayerMLP
from models.three_layer_MLP import ThreeLayerMLP 
from task import classification_engine

NUM_EPOCHS = 50
LEARNING_RATE = 0.1
BATCH_SIZE = 128
# Download dataset
if __name__ == "__main__":
    data_name = r"hojjatk/mnist-dataset"
    data_path = r"./data"
    download_data_and_clear_cache(data_name, data_path)

    #device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Load train data
    train_data = MNISTDataset(
        images_filepath=r"./data/train-images.idx3-ubyte",
        labels_filepath=r"./data/train-labels.idx1-ubyte"
    )
    train_dataloader = DataLoader(train_data, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True)
    # Load test data
    test_data = MNISTDataset(
        images_filepath=r"./data/t10k-images.idx3-ubyte",
        labels_filepath=r"./data/t10k-labels.idx1-ubyte"    
    )
    test_dataloader = DataLoader(test_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    # Some information about dataset
    print(f"Length of train: {len(train_data)}")
    print(f"Length of test: {len(test_data)}")
    image_size = train_data[0][0][0].shape
    num_labels = len(set(train_data.labels))
    print(f"Size sample data train(image, label) at index 0: {image_size}")
    print(f"Number of classes: {num_labels}")

    # Define model
    model_one_layer_MLP = OneLayerMLP(input_shape=image_size[0]*image_size[1], output_shape=num_labels).to(device)
    model_three_layer_MLP = ThreeLayerMLP(input_shape=image_size[0]*image_size[1], output_shape=num_labels).to(device)
    models = {"one_layer_MLP": model_one_layer_MLP, 
              "three_layer_MLP": model_three_layer_MLP}
    
    # Define loss function, optimizer, trainer
    loss_fn = torch.nn.CrossEntropyLoss()
    for model_name, model in models.items():
        print(f"START TRAINING {model_name} MODEL...")
        optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
        trainer = classification_engine.ClassificationTraining(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        results = trainer.train(epochs=NUM_EPOCHS, target_dir="./checkpoints", model_name=model_name + ".pth")
        print(f"DONE TRAINING {model_name} MODEL.")
        print(f"Close plots to continue the training process of the next model...")
        plot_metrics(results, epochs=NUM_EPOCHS, model_name=model_name)
    print("ALL TRAINING PROCESS DONE!")
