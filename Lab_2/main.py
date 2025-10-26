import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
# Import các class cần thiết trực tiếp
from models import LeNet, GoogleNet, ResNet18, ResNet50
from task import classification_engine
from dataloader import MNIST, ViNaFood21, MonkeyDataset
from utils.utils import plot_metrics, collate_fn
from collections import Counter
import argparse # Để đọc tham số dòng lệnh
import yaml    # Để đọc YAML
import warnings

warnings.filterwarnings("ignore", message=".*number of unique classes.*", category=UserWarning)

#  Main Execution 
if __name__ == "__main__":
    #  Parser Tham số Dòng lệnh 
    parser = argparse.ArgumentParser(description="Train a classification model based on a YAML config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    #  Load Configuration 
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Lấy experiment đầu tiên (và duy nhất) từ list 'experiments'
            if 'experiments' not in config or not config['experiments']:
                 raise ValueError("YAML must contain an 'experiments' list with one experiment.")
            exp_config = config['experiments'][0]
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error processing YAML file: {e}")
        exit(1)

    #  Trích xuất thông tin từ Config 
    exp_name = exp_config['name']
    model_key = exp_config['model']       # Tên model (vd: 'ResNet18')
    dataset_key = exp_config['dataset']     # Tên dataset (vd: 'vinafood')
    hp = exp_config['hyperparameters'] # Siêu tham số (vd: lr, batch_size, ...)
    global_settings = config.get('global_settings', {})
    seed = global_settings.get('seed', 42)
    checkpoint_dir = global_settings.get('checkpoint_dir', "./checkpoints")

    print(f"\n Running Experiment: {exp_name} ")
    print(f"Model: {model_key}, Dataset: {dataset_key}")
    print(f"Hyperparameters: {hp}")

    #  Setup 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    #  Map tên model sang Class (Giống code gốc) 
    model_classes = {
        'LeNet': LeNet,
        'GoogleNet': GoogleNet,
        'ResNet18': ResNet18,
        'ResNet50': ResNet50
    }
    if model_key not in model_classes:
        print(f"Error: Model '{model_key}' not found in model_classes mapping.")
        exit(1)
    ModelClass = model_classes[model_key]
    #  Load Dataset 
    print(f"Loading dataset: {dataset_key}")
    try:
        dataset_info = config['datasets'][dataset_key]
        num_classes = dataset_info['classes']
        if dataset_key == 'mnist':
            train_data = MNIST.MNISTDataset(
                images_filepath=dataset_info['train_images'],
                labels_filepath=dataset_info['train_labels']
            )
            test_data = MNIST.MNISTDataset(
                images_filepath=dataset_info['test_images'],
                labels_filepath=dataset_info['test_labels']
            )
        elif dataset_key == 'vinafood':
            train_data = ViNaFood21.ViNaFood21Dataset(path=dataset_info['train_path'], is_train=True)
            test_data = ViNaFood21.ViNaFood21Dataset(path=dataset_info['test_path'], is_train=False)
        elif dataset_key == 'monkey':
            train_data = MonkeyDataset(path=dataset_info['train_path'], is_train=True)
            test_data = MonkeyDataset(path=dataset_info['test_path'], is_train=False)
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
    except KeyError as e:
         print(f"Error accessing dataset info: Missing key {e} in config.")
         exit(1)
    except FileNotFoundError as e:
         print(f"Error loading dataset: File not found - {e}")
         exit(1)

    print(f"Length of train: {len(train_data)}")
    print(f"Length of test: {len(test_data)}")
    image_size = train_data[0]['image'].shape
    num_labels = len(set(train_data.labels))
    print(f"Size sample data train(image, label) at index 0: {image_size}")
    print(f"Number of classes: {num_labels}")

    #  Khởi tạo Model 
    model = ModelClass(in_channels=image_size[0], num_classes=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    #  Training Setup 
    label_counts = Counter(train_data.labels)
    total_samples = len(train_data)
    num_classes = len(label_counts)
    print(f"\nClass Distribution & Weights:")
    print("=" * 70)
    print(f"{'Class ID':<10s} {'Class Name':<30s} {'Count':<10s} {'Weight':<15s}")
    print("=" * 70)

    class_weights = []
    for label_id in train_data.idx2label.keys(): 
        count = label_counts.get(label_id, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)
        class_name = train_data.idx2label[label_id]
        print(f"{label_id:<10d} {class_name:<30s} {count:<10d} {weight:<15.4f}")

    print("=" * 70)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Tạo sample_weights cho từng sample dựa trên class_weights đã tính
    sample_weights = [class_weights[label_id] for label_id in train_data.labels]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=total_samples,
        replacement=True
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    #  DataLoaders 
    batch_size = hp['batch_size']
    num_workers = os.cpu_count() 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Chọn Optimizer
    optimizer_name = hp['optimizer']
    optimizer_params = hp.get('optimizer_params', {})
    learning_rate = hp['lr']
    weight_decay = hp['weight_decay']
    step_size = hp['step_size']
    gamma = hp['gamma']

    print(f"Using optimizer: {optimizer_name} with LR={learning_rate}, WeightDecay={weight_decay}")
    try:
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_params)
        elif optimizer_name.lower() == "sgd":
            if 'momentum' not in optimizer_params:
                optimizer_params['momentum'] = 0.9 # Mặc định
            optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        exit(1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Khởi tạo Trainer
    trainer = classification_engine.ClassificationTraining(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    print(f"START TRAINING {exp_name}...")
    model_filename = f"{exp_name}.pth" # Tên file checkpoint theo tên kịch bản
    checkpoint_path = os.path.join(checkpoint_dir, model_filename)

    # Load checkpoint
    start_epoch = trainer.load_checkpoint(checkpoint_path)
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch} for {exp_name}")

    #  Train 
    results, actual_epochs_ran = trainer.train(
        epochs=hp['epochs'],
        target_dir=checkpoint_dir,
        model_name=model_filename,
        start_epoch=start_epoch
    )
  

    print(f"DONE TRAINING {exp_name}. Ran for {actual_epochs_ran} epochs.")

    #  Plot Results 
    print(f"Plotting results for {exp_name}. Close plot to finish.")
    plot_metrics(results, epochs=actual_epochs_ran, model_name=exp_name)

    print(f"\nExperiment {exp_name} finished successfully!")