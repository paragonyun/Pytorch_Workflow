import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def bert_tokenizer(device="cuda" if torch.cuda.is_available() else "cpu"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased"
    )
    model.to(device)
    print(device)
    return tokenizer, model


def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}
    torch.save(state_dict, save_path)
    print(f"Model is Saved to ==> {save_path}")


def load_checkpoint(
    load_path, model, device="cuda" if torch.cuda.is_available() else "cpu"
):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Model is Loaded <== {load_path}")
    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "global_steps_list": global_steps_list,
    }
    torch.save(state_dict, save_path)
    print(f"Model Metrics are Saved to ==> {save_path}")


def load_metrics(load_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Model Metrics are Loaded <== {load_path}")
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["global_steps_list"],
    )
