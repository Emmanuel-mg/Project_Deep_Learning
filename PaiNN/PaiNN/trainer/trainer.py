from PaiNN.data_loader import PaiNNDataLoader

train_set = PaiNNDataLoader()
val_set = train_set.get_val()
test_set = train_set.get_test()

if __name__=="__main__":
    for i, batch in enumerate(train_set):
        for j, sample in enumerate(batch):
            inputs, targets = sample
            # model(inputs)
            print(targets)