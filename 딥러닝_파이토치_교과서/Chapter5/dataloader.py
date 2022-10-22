from torch.utils.data import Dataset, DataLoader

class FashionDataLoader :
    def __init__(self, train_dataset, test_dataset) :
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def loaders(self) :
        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)

        return train_loader, test_loader

    '''
    loader = FashionDataLoader(train_dataset, test_dataset)
    train_loader, test_loader = loader.loaders()
    '''


