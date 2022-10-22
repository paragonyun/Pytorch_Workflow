'''
Wandb

협업, Coode Versioning, 실험 결과 기록 기능 제공
MLOps의 대표적인 툴!


pip install wandb -q

'''
import wandb 

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

config = {'epochs' : EPOCHS, 
        'batch_size' : BATCH_SIZE,
        'learning_rate' : LEARNING_RATE}

wandb.init(project='test_project', config=config)

for e in range(1, EPOCHS+1) :
    epoch_loss = 0
    epoch_acc = 0
    for x, y in 'dataloader' :

        ...

    wandb.log({'accuracy' : epoch_acc, 'loss' : epoch_loss})




