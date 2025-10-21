import os
import torch
from utils.utils import download_data_and_clear_cache, plot_metrics, collate_fn
from dataloader import MNIST, ViNaFood21 
from torch.utils.data import DataLoader
from models import LeNet, GoogleNet, ResNet18, ResNet50
from task import classification_engine
import kagglehub
import warnings
warnings.filterwarnings("ignore", message=".*number of unique classes.*", category=UserWarning)
#  Hyperparameters 
NUM_EPOCHS = 250
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
EARLY_STOP_EPOCHS = 10

if __name__ == "__main__":
    # Download MNIST dataset
    kagglehub.dataset_download(r'hojjatk/mnist-dataset') # Download if needed

    #  Setup 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    #  Datasets 
    data = {
        'mnist': {
            'train': MNIST.MNISTDataset(
                images_filepath=r"./kaggle/input/mnist-dataset/train-images.idx3-ubyte",
                labels_filepath=r"./kaggle/input/mnist-dataset/train-labels.idx1-ubyte"
            ),
            'test' : MNIST.MNISTDataset(
                images_filepath=r"./kaggle/input/mnist-dataset/t10k-images.idx3-ubyte",
                labels_filepath=r"./kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte"
            ),
            'classes': 10  # Number of classes for MNIST
        },
        'vinafood': {
            'train': ViNaFood21.ViNaFood21Dataset(path=r"/kaggle/input/vinafood21/VinaFood21/train"),
            'test' : ViNaFood21.ViNaFood21Dataset(path=r"/kaggle/input/vinafood21/VinaFood21/test"),
            'classes': 21 # Number of classes for ViNaFood21
        }
    }
    
    #  Model Classes 
    model_classes = {
        'LeNet': LeNet.LeNet,             # Store class, not instance
        'GoogleNet': GoogleNet.GoogleNet,
        'ResNet18': ResNet18.ResNet18,
        'ResNet50': ResNet50.ResNet50
    }

    #  Experiment Configurations 
    configs = [
        {'model_name': 'LeNet', 'data_key': 'mnist'},
        {'model_name': 'GoogleNet', 'data_key': 'vinafood'},
        {'model_name': 'ResNet18', 'data_key': 'vinafood'},
        {'model_name': 'ResNet50', 'data_key': 'vinafood'},
    ]

    #  Training Loop 
    for config in configs:
        model_name = config['model_name']
        data_key = config['data_key']
        
        #  Get Data Info 
        current_data_info = data[data_key] 
        num_classes = current_data_info['classes']
        
        #  Initialize Model 
        print(f"\nInitializing model: {model_name} with {num_classes} classes for {data_key} data")
        ModelClass = model_classes[model_name]
        model = ModelClass(num_classes=num_classes).to(device) # Init model here

        #  DataLoaders 
        train_data = current_data_info['train']
        train_dataloader = DataLoader(
                                    train_data, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True,
                                    collate_fn=collate_fn,
                                )
        test_data = current_data_info['test']
        test_dataloader = DataLoader(
                                    test_data, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False,
                                    collate_fn=collate_fn
                                )
        
    #  Print Data Info 
        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        image_shape = train_data[0]['image'].shape # Get the shape from the tensor image
        print(f"Sample image shape: {image_shape}")
        
        #  Training Setup 
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        trainer = classification_engine.ClassificationTraining(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )        
        print(f"START TRAINING {model_name}...")
        checkpoint_path = f"./checkpoints/{model_name}.pth"
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch} for model {model_name}")
        #  Train 
        results, actual_epochs_ran = trainer.train(epochs=NUM_EPOCHS, 
                                early_stop_epochs = EARLY_STOP_EPOCHS,
                                target_dir="./checkpoints", 
                                model_name=model_name + ".pth",
                                start_epoch=start_epoch)
        
        print(f"DONE TRAINING {model_name}. Ran for {actual_epochs_ran} epochs.")
        # Plot Results
        print(f"Plotting results for {model_name}. Close plot to continue...")
        plot_metrics(results, epochs=actual_epochs_ran, model_name=model_name)

    print("\nALL TRAINING PROCESSES DONE!")