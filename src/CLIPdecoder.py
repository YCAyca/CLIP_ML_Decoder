import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import clip.clip as clip  # OpenAI's CLIP package
import numpy as np
from sklearn.metrics import average_precision_score, recall_score
from PIL import Image
import os
import pathlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
from args import get_arguments

# ---------------------------
# Define your Dataset class
# ---------------------------
def custom_collate_fn(batch):
    # Extract individual elements from the batch
    images = torch.stack([item[0] for item in batch])  # Stack image tensors
    labels = torch.stack([torch.tensor(item[1]) for item in batch])  # Stack label vectors
    metadata = [item[2] for item in batch] if len(batch[0]) > 2 else None  # Optional metadata
    return {'images': images, 'labels': labels, 'metadata': metadata}

def get_train_transform(clip_preprocess):
    """
    Returns a transform for training that includes
    some random augmentations + the default CLIP preprocess.
    """
    return T.Compose([
        # Random horizontal flip with 50% probability
        T.RandomHorizontalFlip(p=0.5),
        # Optional color jitter with a certain probability
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ], p=0.3),
        # Finally, apply the standard CLIP transforms
        # (which includes resizing/cropping to 224, normalizing, etc.)
        clip_preprocess
    ])

def get_val_transform(clip_preprocess):
    """
    Returns a deterministic transform for validation/test
    that uses the default CLIP preprocess (no random augmentations).
    """
    return clip_preprocess



class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None, classnames=[], process="train", incremental=False):
        self.folder_path = folder_path
        self.transform = transform
        self.process = process
        self.incremental = incremental
        self.image_paths = []
        self.labels = []
       
        self.classnames = classnames
        label_dict = {name: i for i, name in enumerate(classnames)}
        self.labels_dict = label_dict
        self.dataset_counter = dict.fromkeys(self.classnames, 0)
        images = os.listdir(os.path.join(folder_path, "images"))
        
        for image_name in images:
            image_labels = []
            self.image_paths.append(os.path.join(folder_path, "images", image_name))
            annotation_file_name = image_name.replace(pathlib.Path(image_name).suffix, ".txt")
            try:
                annotation_file = open(os.path.join(self.folder_path, "labels", annotation_file_name), "r")
            except:
                print("No label file for image: ", image_name, "ignoring image")
                continue
            annotations = annotation_file.readlines()
            for annotation in annotations:
                label = annotation.strip()
                self.dataset_counter[label] += 1
                index = self.classnames.index(label)
                image_labels.append(index)

            # Convert to fixed-length binary vectors
            label_vector = [0] * len(self.classnames)
            for label in image_labels:
                label_vector[label] = 1 
            self.labels.append(label_vector)
  
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image from disk
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ---------------------------
# Define the Decoder Block (Transformer style)
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        """
        A single decoder block with cross-attention and feed forward network.
        """
        super(DecoderBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, query, key, value):
        # query: (num_queries, batch, hidden_dim)
        # key, value: (seq_len, batch, hidden_dim)
        attn_output, _ = self.cross_attn(query, key, value)
        query = query + attn_output
        query = self.norm1(query)
        ff_output = self.ff(query)
        query = query + ff_output
        query = self.norm2(query)
        return query

# ---------------------------
# Updated CLIP + Decoder Model with Group Decoding
# ---------------------------
class CLIPDecoder(nn.Module):
    def __init__(self, num_classes, clip_model, embed_dim, num_groups=4, num_layers=1, num_heads=8, ff_dim=2048, dropout=0.1):
        """
        num_classes: number of labels in your multi-label problem
        clip_model: the loaded CLIP model (will be frozen)
        embed_dim: dimension of the CLIP image features
        num_groups: number of learnable group queries
        num_layers: number of decoder layers
        num_heads: number of attention heads in each decoder layer
        ff_dim: hidden dimension for the feed forward network inside each decoder layer
        """
        super(CLIPDecoder, self).__init__()
        self.clip_model = clip_model
        self.num_groups = num_groups
        self.hidden_dim = embed_dim
        
        # Learnable group queries: shape (num_groups, embed_dim)
        self.group_queries = nn.Parameter(torch.randn(num_groups, embed_dim))
        
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        
        # Final classification head: map from aggregated group features to num_classes logits
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: batch of images
        # Get image features from CLIP (frozen)
        with torch.no_grad():
            features = self.clip_model.encode_image(x)  # shape: (batch, embed_dim)
        # For cross-attention, we treat image features as a sequence with length 1.
        features = features.unsqueeze(0)  # shape: (1, batch, embed_dim)
        
        # Prepare group queries: expand to shape (num_groups, batch, embed_dim)
        batch_size = x.size(0)
        queries = self.group_queries.unsqueeze(1).expand(self.num_groups, batch_size, self.hidden_dim)
        
        # Pass queries through each decoder layer
        for layer in self.decoder_layers:
            queries = layer(queries, features, features)  # cross-attention step
        
        # Group decoding: aggregate the group outputs (here we use a simple mean)
        aggregated = queries.mean(dim=0)  # shape: (batch, embed_dim)
        logits = self.classifier(aggregated)
        return logits

# ---------------------------
# Loss Functions: Classification + Align Loss
# ---------------------------
def compute_align_loss(clip_model, images, targets, text_features, device):
    """
    Computes an alignment loss between the image features and the text features.
    For each positive label in targets, we compute: loss = 1 - cosine_similarity(image_feature, text_feature)
    and average over all positive labels.
    """
    with torch.no_grad():
        image_features = clip_model.encode_image(images)  # shape: (batch, embed_dim)
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)  # shape: (num_classes, embed_dim)
    # Compute cosine similarity between every image and every class text embedding: (batch, num_classes)
    cos_sim = image_features @ text_features_norm.t()
    # For positive labels (where targets == 1), we want cos_sim to be as close to 1 as possible.
    loss_align = torch.sum((1 - cos_sim) * targets) / (targets.sum() + 1e-8)
    return loss_align

# ---------------------------
# Training and Validation Functions with Align Loss
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, clip_model, text_features, device, align_loss_weight=1.0):
    model.train()
    epoch_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        cls_loss = criterion(outputs, targets)
        
        # Compute alignment loss
        loss_align = compute_align_loss(clip_model, images, targets, text_features, device)
        
        total_loss = cls_loss + align_loss_weight * loss_align
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item() * images.size(0)

        # Collect predictions for metrics
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

    epoch_loss /= len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics: average precision and recall with threshold thresh
    avg_precision = average_precision_score(all_targets, all_preds, average='macro')
    binarized_preds = (all_preds > thresh).astype(int)
    avg_recall = recall_score(all_targets, binarized_preds, average='macro', zero_division=0)
    return epoch_loss, avg_precision, avg_recall

def validate(model, loader, criterion, clip_model, text_features, device, align_loss_weight=1.0):
    model.eval()
    epoch_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            cls_loss = criterion(outputs, targets)
            loss_align = compute_align_loss(clip_model, images, targets, text_features, device)
            total_loss = cls_loss + align_loss_weight * loss_align
            epoch_loss += total_loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.detach().cpu().numpy())

    epoch_loss /= len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    binarized_preds = (all_preds > 0.5).astype(int)
    avg_precision = average_precision_score(all_targets, all_preds, average='macro')
    avg_recall = recall_score(all_targets, binarized_preds, average='macro', zero_division=0)
    return epoch_loss, avg_precision, avg_recall

# ---------------------------
# Main Training Loop with Align Loss Setup and Plotting
# ---------------------------
thresh = 0.5

def main_training(num_groups, num_layers, num_heads, ff_dim, dropout):
    os.makedirs(os.path.join(args.output_path,args.exp_name), exist_ok=True)
    model = CLIPDecoder(num_classes, clip_model, clip_model.visual.output_dim, 
                         num_groups=num_groups, num_layers=num_layers, 
                         num_heads=num_heads, ff_dim=ff_dim, dropout=dropout).to(device)
    if args.experimental_run:
        exp_name = "num_groups_"+str(num_groups)+"num_layers_"+str(num_layers)+"num_heads"+str(num_heads)+"ff_dim"+str(ff_dim)+"dropout"+str(dropout)
    else:
        exp_name = args.exp_name
    # Freeze CLIP parameters
    for param in model.clip_model.parameters():
        param.requires_grad = False
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Load datasets
    train_dataset = MultiLabelDataset(folder_path=os.path.join(args.dataset_path, "train"), 
                                      transform=train_preprocess, classnames=class_names)
    val_dataset = MultiLabelDataset(folder_path=os.path.join(args.dataset_path, "test"), 
                                    transform=val_preprocess, classnames=class_names)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    best_mean_acc, counter, best_recall, best_precision = 0.0, 0,0,0
    train_losses, val_losses = [], []
    train_precisions, train_recalls = [], []
    test_precisions, test_recalls = [], []
    
    for epoch in range(args.num_epochs):
        train_loss, train_ap, train_recall = train_one_epoch(model, train_loader, optimizer, criterion, 
                                                              clip_model, text_features, device, 1.0)
        val_loss, val_ap, val_recall = validate(model, val_loader, criterion, clip_model, text_features, 
                                                device, 1.0)
        
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Precision: {train_ap:.4f} | Recall: {train_recall:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Precision: {val_ap:.4f} | Recall: {val_recall:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_precisions.append(train_ap)
        train_recalls.append(train_recall)
        test_precisions.append(val_ap)
        test_recalls.append(val_recall)
        
        mean_acc = (val_ap + val_recall) / 2
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_precision = val_ap
            best_recall = val_recall
            model_path = os.path.join(args.output_path, exp_name, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print("Best model saved!")
            counter = 0
        else:
            counter += 1
        
        if counter >= args.early_stop_patience:
            print("Early stopping at epoch", epoch)
            break
        
    # Save plots
    plt.figure()
    plt.plot(range(1, epoch+2), train_losses, label="Train Loss")
    plt.plot(range(1, epoch+2), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(args.output_path, exp_name, "loss_plot.png"))
    plt.close()

    # Plot and save train precision and recall
    plt.figure()
    plt.plot(range(1, epoch+2), train_precisions, label="Train Precision")
    plt.plot(range(1, epoch+2), train_recalls, label="Train Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Train Precision & Recall")
    plt.legend()
    plt.savefig(os.path.join(args.output_path, exp_name, "precision_recall_train.png"))

    plt.close()

    # Plot and save test (validation) precision and recall
    plt.figure()
    plt.plot(range(1, epoch+2), test_precisions, label="Test Precision")
    plt.plot(range(1, epoch+2), test_recalls, label="Test Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Test Precision & Recall")
    text_str = "Best Precision: {:.3f}\nBest Recall: {:.3f}".format(best_precision, best_recall)    
    plt.gcf().text(0.5, 0.02, text_str, fontsize=12, ha="center", va="bottom", bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    plt.legend()
    plt.savefig(os.path.join(args.output_path, exp_name, "precision_recall_tests.png"))
    plt.close()

    return best_precision, best_recall

if __name__ == '__main__':
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load CLIP model
    clip_model, preprocess, _ = clip.load("ViT-B/32", device=device, jit=False)
    train_preprocess = get_train_transform(preprocess) if args.augment else preprocess
    val_preprocess = preprocess
    clip_model.train()  # Do not freeze the CLIP encoder
    
    # Load class names
    class_names = []
    with open(os.path.join(args.dataset_path, "classnames.txt"), 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    num_classes = len(class_names)
    
    # Encode text prompts with CLIP
    class_prompts = [f"{name}" for name in class_names]
    with torch.no_grad():
        text_tokens = clip.tokenize(class_prompts).to(device)
        text_features = clip_model.encode_text(text_tokens)
    

    # Initialize model
    if args.experimental_run:
        ff_dims = [512, 768, 1024, 2048]
        num_heads = [1,2,4,8]
        drops = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        num_layers = [1,2]
        best_best_recall = 0
        best_best_precision = 0
        for i in range(1,12): #num groups
                for k in num_heads: #num heads
                    for d in drops:  # drop out
                        for ff_dim in ff_dims:
                            for n_layer in num_layers:
                                best_precision,best_recall= main_training(num_groups=i, num_layers=n_layer, num_heads=k, ff_dim=ff_dim, dropout=d)
                                if best_precision > best_best_precision:
                                    best_best_precision = best_precision
                                    parameters_prec = "num groups " + str(i) + "num heads " + str(k) + "drop " + str(d) + "ff dim " + str(ff_dim)
                                if best_recall > best_best_recall:
                                    best_best_recall = best_recall
                                    parameters_recall = "num groups " + str(i) + "num heads " + str(k) + "drop " + str(d) + "ff dim " + str(ff_dim)


                                print("BEST PRECISION OVER GRID SEARCH", best_best_precision)
                                print("BEST RECALL  OVER GRID SEARCH", best_best_recall)

                                print("BEST PARAMETERS PRECISION  OVER GRID SEARCH", parameters_prec)
                                print("BEST PARAMETERS RECALL  OVER GRID SEARCH", parameters_recall)
    else:
        best_precision,best_recall= main_training(num_groups=args.num_groups, num_layers=args.num_layers, num_heads=args.num_heads, ff_dim=args.ff_dim, dropout=args.dropout)
