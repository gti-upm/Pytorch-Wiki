# Pytorch-Wiki

# Dataset
- Create a custom Dataset
```python
class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images for autoencoder.
    Args:
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image) # At least: T.ToTensor()

        return tensor_image, tensor_image
```
- Split a dataset
```python
train_size = train_size = int(TRAIN_RATIO * len(full_dataset)) #full_dataset is a Dataset objecct
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
```

# Dataloader
- Train and validation

```python
train_loader = torch.utils.data.DataLoader(    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.TEST_BATCH_SIZE
)
```
# GPU
- Select GPU if available
```python
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"if torch.cuda.is_available():   
```
- Load model into GPU if available
```python
if torch.cuda.is_available():
    print("GPU Availaible moving model to GPU")
else:
    print("Moving model to CPU")
model.to(device)
```

# Initialization
- Seed everything
```python
def seed_everything(seed):
    """
    Makes code deterministic using a given seed. Internally sets all seeds of torch, numpy and random.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

```

# Training
- Custom training and validation step
```python
def train_step(model, train_loader, loss_fn, optimizer, device):
    """
    Performs a single training step
    Args:
    model: torch model
    train_loader: PyTorch dataloader
    loss_fn: PyTorch loss_fn.
    optimizer: PyTorch optimizer.
    device: "cuda" or "cpu"

    Returns: Train Loss
    """
    model.train()

    for batch_idx, (train_img, target) in enumerate(train_loader):
        train_img = train_img.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(train_img)

        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

    return loss.item()


def val_step(model, val_loader, loss_fn, device):
    """
    Performs a single training step
    Args:
    model:torch_model 
    val_loader: PyTorch dataloader.
    loss_fn: PyTorch loss_fn.
    device: "cuda" or "cpu"

    Returns: Validation Loss
    """

    model.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target) in enumerate(val_loader):
            train_img = train_img.to(device)
            target = target.to(device)

            output = model(train_img)

            loss = loss_fn(output, target)

    return loss.item()

max_loss = val_loss = 9999
for epoch in tqdm(range(config.EPOCHS)):
    train_loss = train_step(model, train_loader, loss_fn, optimizer, device=device)
    print(f"Epochs = {epoch}, Training Loss : {train_loss}")
    
    val_loss = val_step(model, val_loader, loss_fn, device=device)

    # Simple Best Model saving
    if val_loss < max_loss:
        print("Validation Loss decreased, saving new best model")
        torch.save(model.state_dict(), config.MODEL_PATH)
        max_loss = val_loss

    print(f"Epochs = {epoch}, Validation Loss : {val_loss}")

```