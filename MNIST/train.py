import os
import torch
import dataloader, dataset, model, trainer, load_config

params = {
    ## TODO
}

dataset = dataset.CustomDataset()
train_dataset, test_dataset = dataset._load_fashion_mnist()

dataloader = dataloader.CustomDataLoader(dataset = train_dataset,
                                        batch_size = 32,
                                        val_ratio = 0.2)

train_loader, val_loader = dataloader.split_validation()



cls_model = model.CustomModel()

## optimizer 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cls_model.parameters(),
                            lr = 0.01 ) ## parameter 조정하기
config = load_config.load_config('./config.yaml')

TRAIN = trainer.CustomTrainer(model = cls_model, 
                        criterion=criterion,
                        optimizer=optimizer,
                        config = config
                        )


TRAIN.train(train_dataloader = train_loader, val_dataloader=val_loader)


# if __name__ == '__main__' :
#     main()