import torch
import clip.clip as clip  # OpenAI's CLIP package
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from src.CLIPdecoder import CLIPDecoder
from args import get_arguments  # Importing args from args.py

def load_trained_model(checkpoint_path, num_classes, device, num_groups, num_layers, num_heads, ff_dim, dropout):
    # Load CLIP model and preprocessing transform
    clip_model, preprocess, _  = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    embed_dim = clip_model.visual.output_dim
    
    # Initialize our CLIP+Decoder model and load checkpoint
    model = CLIPDecoder(num_classes, clip_model, embed_dim,
                        num_groups=num_groups, num_layers=num_layers, 
                        num_heads=num_heads, ff_dim=ff_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, preprocess

def infer_image(model, preprocess, image_path, device, threshold):
    # Open image and apply CLIP preprocessing
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = (probs > threshold).astype(int)
    
    return predictions, probs, image

def plot_inference(image, probs, predictions, class_names, save_path):
    """
    Plot the input image with predicted classes and prediction scores.
    Positive predictions are highlighted.
    """
    pred_info = [f"{class_names[idx]}: {score:.2f}" for idx, (pred, score) in enumerate(zip(predictions, probs)) if pred]
    text_str = "\n".join(pred_info) if pred_info else "No predictions above threshold"
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.gcf().text(0.5, 0.02, text_str, fontsize=12, ha="center", va="bottom",
                   bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    args = get_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output folder
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path,"inference"), exist_ok=True)
    
    # Load class names
    class_names = []
    with open(args.classnames, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
    num_classes = len(class_names)
    
    # Load model and preprocess
    model, preprocess = load_trained_model(args.checkpoint_path, num_classes, device, 
                                           args.num_groups, args.num_layers, args.num_heads, 
                                           args.ff_dim, args.dropout)
    
    # Iterate over images in the input folder and perform inference
    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in image_files[:15]:
        image_path = os.path.join(args.input_folder, img_name)
        predictions, probs, image = infer_image(model, preprocess, image_path, device, args.threshold)
        print(f"Inference for {img_name}:")
               
        # Save the plot with the predicted classes
        save_path = os.path.join(args.output_path, args.exp_name, f"{os.path.splitext(img_name)[0]}.png")
        plot_inference(image, probs, predictions, class_names, save_path)
