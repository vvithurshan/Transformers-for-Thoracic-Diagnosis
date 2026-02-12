import os
import itertools
import traceback
import math
import numpy as np
import pickle
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import timm
import wandb

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


columns = ['Enlarged Cardiomediastinum', 
            'Cardiomegaly', 
            'Lung Opacity', 
            'Lung Lesion', 
            'Edema', 
            'Consolidation', 
            'Atelectasis', 
            'Pneumothorax', 
            'Pleural Effusion', 
            'Support Devices']
# --------------------------
# CSV processing (keeps your paths)
# --------------------------
def CSV_Processing(train_path: str, valid_path: str):
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)

    columns = ['Path',
                'Enlarged Cardiomediastinum', 
                'Cardiomegaly', 
                'Lung Opacity', 
                'Lung Lesion', 
                'Edema', 
                'Consolidation', 
                'Atelectasis', 
                'Pneumothorax', 
                'Pleural Effusion', 
                'Support Devices']
    
    # select only the columns we need
    df_train = df_train[columns]
    df_valid = df_valid[columns]
    
    # fill missing values with 0
    df_train = df_train.fillna(0)
    df_valid = df_valid.fillna(0)
    
    # drop rows with negative values
    df_train = df_train[(df_train[columns[1:]] >= 0).all(axis=1)]
    df_valid = df_valid[(df_valid[columns[1:]] >= 0).all(axis=1)]


    # fraction of data
    # df_train = df_train.sample(frac=0.01)
    
    
    # fraction of data
    # df_train = df_train.sample(frac=0.01)
    
    # drop raws with frontal view in path column
    df_train = df_train[df_train['Path'].str.contains('frontal')]
    df_valid = df_valid[df_valid['Path'].str.contains('frontal')]

    return df_train, df_valid

train_df, valid_df = CSV_Processing('../Trial-1/CheXpert-v1.0-small/train.csv', '../Trial-1/CheXpert-v1.0-small/valid.csv')


# create tuple labels used earlier for weighted sampler / rare augmentation logic
train_df['label_tuple'] = train_df[columns].apply(tuple, axis=1)
valid_df['label_tuple'] = valid_df[columns].apply(tuple, axis=1)

label_freq = train_df['label_tuple'].value_counts().to_dict()
print("Label frequencies (train):", label_freq)

sample_weights = [1.0 / label_freq[t] for t in train_df['label_tuple']]

threshold = 0.05 * len(train_df)
RARE_CLASSES = set([label for label, count in label_freq.items() if count < threshold])
print("Rare label combinations:", RARE_CLASSES)


# --------------------------
# Transforms (medical-safe)
# --------------------------
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
rare_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),    
])


# --------------------------
# Multi-label Dataset
# --------------------------
class MultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform_normal, transform_rare, rare_classes, root_prefix="../Trial-1/"):
        self.paths = root_prefix + df["Path"].values
        self.labels = df[columns].values.astype(np.float32)
        self.label_tuples = df['label_tuple'].values
        self.transform_normal = transform_normal
        self.transform_rare = transform_rare
        self.rare_classes = rare_classes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label_tuple = tuple(self.label_tuples[idx])
        label = self.labels[idx]

        if label_tuple in self.rare_classes:
            img = self.transform_rare(img)
        else:
            img = self.transform_normal(img)

        return img, torch.from_numpy(label)




# --------------------------
# Metrics helpers (multi-label)
# --------------------------
def sigmoid(x: np.ndarray):
    # Convert numpy array back to tensor for torch.sigmoid()
    return torch.sigmoid(torch.from_numpy(x))


def compute_multilabel_metrics(y_true: np.ndarray, y_logits: np.ndarray, threshold=0.5):
    """
    y_true: (N, C) binary
    y_logits: (N, C) raw logits (NumPy array)
    """
    # y_logits must be a NumPy array here, which is passed to sigmoid(x)
    y_probs = sigmoid(y_logits).numpy() # Convert back to NumPy after sigmoid for sklearn metrics
    y_pred = (y_probs >= threshold).astype(int)

    # per-class precision/recall/f1
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    # micro f1
    _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    # per-class AUC (may fail if a class has only one label in val set)
    aucs = []
    C = y_true.shape[1]
    for c in range(C):
        try:
            auc = roc_auc_score(y_true[:, c], y_probs[:, c]) 
        except Exception:
            auc = np.nan
        aucs.append(auc)

    # 👉 added: macro AUC (ignore NaNs when averaging)
    auc_macro = np.nanmean(aucs)

    # example-based (subset) accuracy (all labels match)
    exact_acc = np.mean((y_pred == y_true).all(axis=1).astype(float))

    return {
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "f1_micro": f1_micro,
        "aucs": aucs,               # per-class AUC
        "auc_macro": auc_macro,     # 👉 added macro AUC
        "exact_match_acc": exact_acc,
        "y_probs": y_probs,
        "y_pred": y_pred
    }


def plot_per_class_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    C = y_true.shape[1]

    assert len(class_names) == C, f"class_names({len(class_names)}) != num_classes({C})"

    cols = 3
    rows = math.ceil(C / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.atleast_2d(axs)

    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        ax = axs[r, c]

        if i < C:
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(class_names[i])
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")

            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    ax.text(jj, ii, str(cm[ii, jj]),
                            ha="center", va="center",
                            color="white" if cm[ii, jj] > cm.max() / 2 else "black")
        else:
            ax.axis('off')

    fig.tight_layout()
    return fig



# --------------------------
# Training / Validation loops
# --------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, scheduler=None):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0

    all_logits = []
    all_labels = []

    # Determine device type for torch.amp.autocast
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        # Using torch.amp.autocast to avoid FutureWarning
        with torch.amp.autocast(device_type=device_type, enabled=(device != "cpu")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            # OneCycleLR expects step() per batch
            try:
                scheduler.step()
            except Exception:
                pass

        running_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_multilabel_metrics(all_labels, all_logits, threshold=0.5)
    return epoch_loss, metrics, all_labels, all_logits


def validate_one_epoch(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_multilabel_metrics(all_labels, all_logits, threshold=0.5)
    return epoch_loss, metrics, all_labels, all_logits


# --------------------------
# Main experiment loop
# --------------------------
def main():
    # Master dictionary to store results for all experiments
    ALL_RESULTS_HISTORY: Dict[str, Any] = {}
    
    # hyperparameter grid (running first element of each list for quick test/debug)
    model_names = [
        "vit_base_r50_s16_224",
        "vit_large_r50_s32_224",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window7_224",
        "deit_base_patch16_224",
    ]

    learning_rates = [2e-4]
    batch_sizes = [64]
    
    # --- EARLY STOPPING CONFIG ---
    PATIENCE = 5
    # -----------------------------

    # --- Slicing to run only the first combination for quick testing ---
    # REMOVE these lines to run the full sweep:
    # model_names = model_names[5:6]
    # lora_r_values = lora_r_values[0:1]
    # lora_dropouts = lora_dropouts[0:1]
    # learning_rates = learning_rates[0:1]
    # batch_sizes = batch_sizes[0:1]
    # -------------------------------------------------------------------

    EPOCHS = 20
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Debug Check for TypeError ---
    if not isinstance(model_names, list) or \
       not isinstance(learning_rates, list) or \
       not isinstance(batch_sizes, list):
        raise TypeError("One or more hyperparameter variables is not a list. Check definitions in main().")
    # --- End Debug Check ---

    # CSV dataset paths (keep as you had them)
    train_csv = '../Trial-1/CheXpert-v1.0-small/train.csv'
    valid_csv = '../Trial-1/CheXpert-v1.0-small/valid.csv'

    # Build datasets once (we'll re-create loaders per batch size)
    # Note: we already loaded train_df, valid_df above

    total_runs = len(model_names) * len(learning_rates) * len(batch_sizes)
    print(f"Total experiment runs: {total_runs}")

    combinations = list(itertools.product(model_names, learning_rates, batch_sizes))

    for model_name, lr, bs in combinations:
        exp_name = f"{model_name}_lr{lr}_bs{bs}"
        print("\n" + "="*100)
        print(f"Starting experiment: {exp_name}")
        print(f"Model: {model_name} | lr: {lr} | bs: {bs}")
        print(f"Early Stopping Patience: {PATIENCE} epochs")
        print("="*100)

        # Initialize evaluation history for the current experiment
        evaluation_history: Dict[str, List[Dict[str, Any]]] = {
            "val_metrics": []
        }
        
        # Early Stopping Counters
        best_val_f1 = -1
        epochs_no_improve = 0
        should_stop = False

        try:
            # WandB + TB RE-ENABLED
            wandb.login(key=os.environ["WANDB_API_KEY"])
            wandb.init(project="vit-PT-Nov-28-last-head-only", name=exp_name, reinit=True)
            wandb.config.update({
                "model_name": model_name,
                "lr": lr,
                "batch_size": bs,
                "epochs": EPOCHS,
                "patience": PATIENCE
            })
            tb_writer = SummaryWriter(log_dir=os.path.join("runs", exp_name))

            # Data loaders (re-create to reflect batch size)
            train_dataset = MultiLabelDataset(train_df, normal_transform, rare_transform, RARE_CLASSES, root_prefix="../Trial-1/")
            val_dataset = MultiLabelDataset(valid_df, normal_transform, rare_transform, RARE_CLASSES, root_prefix="../Trial-1/")

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

            steps_per_epoch = len(train_loader)
            print(f"Steps per epoch: {steps_per_epoch}")

            # Model creation with try/except
            try:
                model = timm.create_model(model_name, pretrained=True, num_classes=10)
            except Exception as e:
                print(f"Model {model_name} cannot be created: {e}")
                wandb.log({"error_create_model": str(e)})
                wandb.finish()
                tb_writer.close()
                continue

            # freeze all parameters
            for param in model.parameters():
                param.requires_grad = False

            # unfreeze previous block (e.g., block 11)
            # for param in model.blocks[11].parameters():
            #     param.requires_grad = True
            
            # for param in model.norm.parameters():
            #     param.requires_grad = True

            # unfreeze last layer
            for param in model.head.parameters():
                param.requires_grad = True


            # Print trainable parameters count
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {num_total_params}")
            print(f"Trainable parameters: {num_trainable_params} ({(num_trainable_params/num_total_params)*100:.2f}%)")
            
            # print model and trainable status
            for name, param in model.named_parameters():
                print(f"{name}: {param.requires_grad}")


            # DataParallel if multiple GPUs
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model)

            model = model.to(device)

            # Optimizer + scheduler (OneCycleLR) - step per batch
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                      steps_per_epoch=steps_per_epoch,
                                                      epochs=EPOCHS)

            # AMP scaler
            scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))

            # Training loop
            best_model_path = None

            for epoch in range(1, EPOCHS + 1):
                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch}. Patience exceeded.")
                    break
                
                train_loss, train_metrics, train_y, train_logits = train_one_epoch(model, train_loader, optimizer, device, scaler, scheduler)
                val_loss, val_metrics, val_y, val_logits = validate_one_epoch(model, val_loader, device)
                
                current_val_f1 = val_metrics["f1_macro"]

                # Log scalar metrics to WandB and TB
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_f1_macro": train_metrics["f1_macro"],
                    "val_f1_macro": current_val_f1,
                    "train_f1_micro": train_metrics["f1_micro"],
                    "val_f1_micro": val_metrics["f1_micro"],
                    "train_exact_acc": train_metrics["exact_match_acc"],
                    "val_exact_acc": val_metrics["exact_match_acc"],
                    "train_auc_macro": train_metrics["auc_macro"],
                    "val_auc_macro": val_metrics["auc_macro"],
                    "lr": optimizer.param_groups[0]["lr"]
                }
                # Add per-class AUCs
                val_auc_dict = {}
                for i, auc in enumerate(val_metrics["aucs"]):
                    log_dict[f"val_auc_class_{i}"] = auc if not np.isnan(auc) else None
                    val_auc_dict[f"val_auc_class_{i}"] = auc if not np.isnan(auc) else None # for history dump

                wandb.log(log_dict, step=epoch)
                # TensorBoard
                tb_writer.add_scalar("train/loss", train_loss, epoch)
                tb_writer.add_scalar("val/loss", val_loss, epoch)
                tb_writer.add_scalar("train/f1_macro", train_metrics["f1_macro"], epoch)
                tb_writer.add_scalar("val/f1_macro", current_val_f1, epoch)
                tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                
                # STORE EVALUATION METRICS FOR DUMP
                epoch_metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_f1_macro": current_val_f1,
                    "val_exact_acc": val_metrics["exact_match_acc"],
                    **val_auc_dict # include per-class AUCs
                }
                evaluation_history["val_metrics"].append(epoch_metrics)


                # Output key metrics to console
                print(f"[{exp_name}] Epoch {epoch}/{EPOCHS} | Train loss {train_loss:.4f} val loss {val_loss:.4f}")
                print(f"  Train F1 macro: {train_metrics['f1_macro']:.4f}, Train Acc: {train_metrics['exact_match_acc']:.4f}")
                print(f"  Val F1 macro: {current_val_f1:.4f}, Val Acc: {val_metrics['exact_match_acc']:.4f}")

                # Confusion matrices per-class
                try:
                    cm_fig = plot_per_class_confusions(val_y, val_metrics["y_pred"], columns)
                    wandb.log({"confusion_matrix": wandb.Image(cm_fig)}, step=epoch)
                    tb_writer.add_figure("confusion_matrix", cm_fig, epoch)
                    plt.close(cm_fig)
                except Exception as e:
                    print("Failed to plot/log confusion matrix:", e)

                # --- EARLY STOPPING CHECK ---
                if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1
                    best_model_path = f"best_{exp_name}.pth"
                    # Save model checkpoint
                    torch.save(model.state_dict(), best_model_path)
                    wandb.save(best_model_path)
                #     epochs_no_improve = 0
                # else:
                #     epochs_no_improve += 1
                #     if epochs_no_improve == PATIENCE:
                #         should_stop = True
                        
                # print(f"  Patience: {epochs_no_improve}/{PATIENCE}")
                # ----------------------------

            print(f"Experiment {exp_name} finished. Best val F1 macro: {best_val_f1:.4f}")
            
            # FINAL DUMP OF EVALUATION HISTORY FOR THIS RUN
            final_dump = evaluation_history["val_metrics"]
            
            # AGGREGATE RESULTS TO MASTER DICTIONARY
            ALL_RESULTS_HISTORY[exp_name] = final_dump
            
            wandb.log({"best_val_f1_macro": best_val_f1, "epochs_ran": epoch})
            wandb.run.summary["evaluation_history"] = final_dump # Store in WandB summary
            
            tb_writer.close()
            wandb.finish()

        except Exception as e:
            print(f"Experiment {exp_name} failed with exception:")
            traceback.print_exc()
            try:
                wandb.finish()
            except Exception:
                pass
            continue
            
    # ----------------------------------------------------
    ## FINAL PICKLE DUMP AFTER ALL EXPERIMENTS ARE COMPLETE
    # ----------------------------------------------------
    import datetime
    pickle_filename = f"all_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    try:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(ALL_RESULTS_HISTORY, f)
        print(f"\nSuccessfully dumped all {len(ALL_RESULTS_HISTORY)} experiment results to {pickle_filename}")
    except Exception as e:
        print(f"\nERROR: Failed to save pickle file {pickle_filename}: {e}")


if __name__ == "__main__":
    main()