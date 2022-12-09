import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from tqdm import tqdm


tokenizer, _ = bert_tokenizer()


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion=nn.BCELoss(),
    num_epochs=100,
    best_valid_loss=float("Inf"),
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    total_correct = 0.0
    total_len = 0.0
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    model.train()
    print("🚀Start Training🚀")
    print(f"Resource : {device}")
    print("Model Structure")
    print(model)

    for epoch in range(num_epochs):
        for text, label in tqdm(train_loader):
            optimizer.zero_grad()
            encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padding_list = [e + [0] * (512 - len(e)) for e in encoded_list]

            sample = torch.tensor(padding_list)
            sample, label = sample.to(device), label.to(device)
            labels = torch.tensor(label)
            outputs = model(sample, labels=labels)
            loss, logits = outputs

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)

            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10000 == 0:  # iter 10000마다 모델 validation
                model.eval()
                with torch.no_grad():
                    for text, label in val_loader:
                        encoded_list = [
                            tokenizer.encode(t, add_special_tokens=True) for t in text
                        ]
                        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
                        sample = torch.tensor(padded_list)

                        sample, label = sample.to(device), label.to(device)

                        outputs = model(sample, lables=labels)
                        loss, logits = outputs
                        valid_running_loss += loss.item()

        average_train_loss = running_loss / 10000
        average_valid_loss = valid_running_loss / len(val_loader)
        train_loss_list.append(average_train_loss)
        valid_loss_list.append(average_valid_loss)
        global_steps_list.append(global_step)

        running_loss = 0.0
        valid_running_loss = 0.0
        model.train()

        print(
            "==================================================================================="
        )
        print(
            f"[EPOCH {epoch+1}/{num_epochs} STEP {global_step}/{num_epochs*len(train_loader)}]"
        )
        print(
            f"Train Loss : {average_train_loss:.4f}   Valid Loss : {average_valid_loss:.4f}"
        )

        if best_valid_loss > average_valid_loss:
            best_valid_loss = average_valid_loss
            save_checkpoint("./model.pt", model, best_valid_loss)
            save_metrics(
                "./metrics.pt", train_loss_list, valid_loss_list, global_steps_list
            )

    save_metrics(
        "./fin_metrics.pt", train_loss_list, valid_loss_list, global_steps_list
    )
    print("🚩Training is Finished🚩")
