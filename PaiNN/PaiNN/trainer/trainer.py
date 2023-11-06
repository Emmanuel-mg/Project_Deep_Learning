from PaiNN.data_loader import PaiNNDataLoader

train_set = PaiNNDataLoader(batch_size=2)
val_set = train_set.get_val()
test_set = train_set.get_test()

if __name__=="__main__":
    for i, batch in enumerate(train_set):
        print(batch)