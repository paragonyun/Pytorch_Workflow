'''
학습 중간중간 최선의 결과 저장

epoch, loss, metric 모두 활용하여 저장!
'''

"""
예시코드

torch.save({
    'epoch' : e,
    'model_state_dict' : model.state_dict(),
    'optimzier_state_dict' : optimizer.state_dict(),
    'loss' : epoch_loss
    },
    f'saved/checkpoint_model_{e}_{epoch_loss/len(dataloader):.2f}_{epoch_acc/len(dataloader):.2f}.pt'
)

checkpoint = torch.load(PATH)
model.load_state_dic(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
"""

