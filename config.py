class Config:
    model_name = 'AlexNet_CIFAR10'
    '''
    model_names:
        'LeNet'
        'Dropout_LeNet'
        'AlexNet_CIFAR10'
        'VGG11_CIFAR10'
    '''
    batch_size = 128
    epochs = 100
    dataset_path = 'D:/CIFAR_dataset'

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    dropout_rate = 0.5

    factor = 0.5
    patience = 5
    threshold = 1e-4

    early_stopping_patience = 10 # 衰减器的耐心不要大于早停耐心