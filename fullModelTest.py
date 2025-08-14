import os
import cv2
from ultralytics import YOLO
import os, time, math, argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import timm as timm
CLASS_NAMES = ["cider","wheat","lager","ipa","stout","not_beer"]

def detect_bounding_boxes_on_image(image_path):
    """
    Prompts the user for an image filename, runs YOLO detection,
    draws bounding boxes and labels on the image,
    and returns a list of bounding boxes found in the image.
    Each bounding box is a dict with keys: x1, y1, x2, y2, label, confidence.
    """
    # Load pretrained model
    model = YOLO('yolov8n.pt')



    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return []

    # Run YOLO detection
    results = model(image_path)

    bounding_boxes = []
    for result in results:
        detections = result.to_df()
        print(f"Found {len(detections)} detections in {image_path}")
        for idx, row in detections.iterrows():
            box = row['box']
            x1 = int(box['x1'])
            y1 = int(box['y1'])
            x2 = int(box['x2'])
            y2 = int(box['y2'])
            label = row['name']
            confidence = float(row['confidence'])
            bounding_boxes.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'label': label,
                'confidence': confidence
            })
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Draw label and confidence
            label_text = f"{label} {confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return bounding_boxes


class BeerClassifier(nn.Module):
    def __init__(self, dropout=0.2, num_classes=6):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def efficient_net_full_train(dropout=0.4, num_classes=6):
    model = timm.create_model("efficientnet_b0", pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model
def efficient_net_feature_extract(dropout=0.4, num_classes=6):
    model = timm.create_model("efficientnet_b0", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    for param in model.classifier.parameters():
        param.requires_grad = True  # only train classifier
    return model
def efficient_net_partial(dropout=0.4, num_classes=6):
    model = timm.create_model("efficientnet_b0", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # freeze everything first
    # Unfreeze last two blocks and classifier
    for name, param in model.named_parameters():
        if "blocks.5" in name or "blocks.6" in name or "classifier" in name:
            param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model
def dino_feature_extract(dropout=0.4, num_classes=6):
    model = timm.create_model("vit_base_patch16_224_dino", pretrained=True, num_classes=0)

    for param in model.parameters():
        param.requires_grad = False

    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.num_features, num_classes)
    )

    return model
def dino_fulltrain(dropout=0.4, num_classes=6):
    model = timm.create_model("vit_base_patch16_224_dino", pretrained=True, num_classes=0)

    for param in model.parameters():
        param.requires_grad = True

    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.num_features, num_classes)
    )

    return model
def dino_partialtrain(dropout=0.4, num_classes=6, n_blocks=3):
    model = timm.create_model("vit_base_patch16_224_dino", pretrained=True, num_classes=0)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last n_blocks
    for name, param in model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(12 - n_blocks, 12)) or "norm" in name:
            param.requires_grad = True

    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.num_features, num_classes)
    )

    return model

if __name__ == "__main__":
    # Request input of image filename (with path if not in current dir)
    image_path = input("Enter the path to the image file: ").strip()
    boxes = detect_bounding_boxes_on_image(image_path)
    img = cv2.imread(image_path)

    CUP_CLASSES = ['cup', 'wine glass']
    BEER_TYPES = CLASS_NAMES

    # Path to save all output cup crops
    OUT_CUPS_DIR = "all_cup_crops"
    os.makedirs(OUT_CUPS_DIR, exist_ok=True)

    # Load multiple classifier models once (ensemble)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_weights(model: nn.Module, ckpt_path: str) -> nn.Module:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and any(k in state for k in ["state_dict", "model_state_dict"]):
            state = state.get("state_dict", state.get("model_state_dict"))
        if hasattr(state, 'state_dict') and callable(getattr(state, 'state_dict')):
            state = state.state_dict()
        if isinstance(state, dict):
            normalized_state = {}
            for k, v in state.items():
                new_k = k
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                if new_k.startswith('model.'):
                    new_k = new_k[len('model.'):]
                normalized_state[new_k] = v
            state = normalized_state
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            model.load_state_dict(state, strict=False)
        return model

    # Configure which models to use and their checkpoints
    MODEL_CONFIGS = {
        # Simple CNN
        "cnn_simple": {
            "factory": lambda: BeerClassifier(dropout=0.2, num_classes=6),
            "ckpt": r"models/BeerClassifier.pth",
        },
        # EfficientNet variants
        "efficientnet_full": {
            "factory": lambda: efficient_net_full_train(dropout=0.188, num_classes=6),
            "ckpt": r"models/best_model_eff_fulltrain.pth",
        },
        "efficientnet_feature": {
            "factory": lambda: efficient_net_feature_extract(dropout=0.109, num_classes=6),
            "ckpt": r"models/best_model_eff_freezeout.pth",
        },
        "efficientnet_partial": {
            "factory": lambda: efficient_net_partial(dropout=0.402, num_classes=6),
            "ckpt": r"models/best_model_eff_partial.pth",
        },
        # DINO ViT variants
        "dino_feature": {
            "factory": lambda: dino_feature_extract(dropout=0.468, num_classes=6),
            "ckpt": r"models/best_model_dino_freezeout.pth",
        },
        "dino_full": {
            "factory": lambda: dino_fulltrain(dropout=0.211, num_classes=6),
            "ckpt": r"models/dino_fulltrain.pth",
        },
        "dino_partial": {
            "factory": lambda: dino_partialtrain(dropout=0.337, num_classes=6, n_blocks=3),
            "ckpt": r"models/best_model_dino_partialtrain.pth",
        },
    }

    loaded_models = []  # list of (name, model)
    for name, cfg in MODEL_CONFIGS.items():
        try:
            mdl = cfg["factory"]().to(device)
            mdl = load_weights(mdl, cfg["ckpt"]).to(device)
            mdl.eval()
            loaded_models.append((name, mdl))
        except Exception as e:
            print(f"[WARN] Could not load {name} from {cfg['ckpt']}: {e}")
    if not loaded_models:
        raise RuntimeError("No classifier models loaded. Check checkpoint paths in MODEL_CONFIGS.")

    # Preprocessing for classifier
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def classify_with_single_model(cup_img_bgr, mdl):
        """Classify a cropped cup image with a single model. Returns (label, confidence)."""
        cup_img_rgb = cv2.cvtColor(cup_img_bgr, cv2.COLOR_BGR2RGB)
        tensor = preprocess(cup_img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = mdl(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        return BEER_TYPES[int(idx.item())], float(conf.item())

    def annotate_with_model(image_bgr, boxes, mdl, model_name):
        annotated = image_bgr.copy()
        # Add small header with model name
        header_h = 28
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], header_h), (50, 50, 50), -1)
        cv2.putText(annotated, f"Model: {model_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for box in boxes:
            h, w = annotated.shape[:2]
            x1 = max(0, min(w - 1, box['x1']))
            y1 = max(0, min(h - 1, box['y1']))
            x2 = max(0, min(w, box['x2']))
            y2 = max(0, min(h, box['y2']))
            if x2 <= x1 or y2 <= y1:
                continue

            # Only show beer annotations for cups; skip other YOLO classes
            if box['label'] not in CUP_CLASSES:
                continue

            crop = image_bgr[y1:y2, x1:x2]
            beer_type, conf = classify_with_single_model(crop, mdl)
            label_text = f"{beer_type} {int(conf*100)}%"
            label_color = (255, 0, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), label_color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text_top = max(header_h, y1 - text_h - baseline)
            cv2.rectangle(annotated, (x1, y_text_top), (x1 + text_w, y_text_top + text_h + baseline), label_color, -1)
            cv2.putText(annotated, label_text, (x1, y_text_top + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return annotated

    def make_grid(panels, cols=None, bg_color=(255, 255, 255)):
        n = len(panels)
        if n == 0:
            return None
        h, w = panels[0].shape[:2]
        if cols is None:
            cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        grid = np.full((rows * h, cols * w, 3), bg_color, dtype=np.uint8)
        for i, panel in enumerate(panels):
            r = i // cols
            c = i % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = panel
        return grid

    # Per-model annotated images
    per_model_images = []  # list of (image_bgr, model_name)
    base, ext = os.path.splitext(image_path)
    for name, mdl in loaded_models:
        ann = annotate_with_model(img, boxes, mdl, name)
        out_path = f"{base}_{name}_annotated{ext if ext else '.jpg'}"
        cv2.imwrite(out_path, ann)
        print(f"Saved: {out_path}")
        per_model_images.append((ann, name))

    # Composite grid image with matplotlib (resizes nicely and adds titles)
    if per_model_images:
        n = len(per_model_images)
        cols = min(3, n)
        rows = int(math.ceil(n / cols))
        fig_w, fig_h = cols * 5, rows * 5  # tune sizing
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(rows, cols) if n > 1 else np.array([[axes]])
        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r, c]
                if idx < n:
                    img_bgr, name = per_model_images[idx]
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.set_title(name, fontsize=12)
                ax.axis('off')
                idx += 1
        plt.tight_layout()
        composite_path = f"{base}_all_models.jpg"
        fig.savefig(composite_path, bbox_inches='tight', dpi=150)
        print(f"Saved composite: {composite_path}")
        plt.show()

