class Config:
    dataset_path = 'D:/CIFAR_dataset'
    model_name = 'VGG11_CIFAR10'
    '''
    model_names:
        'LeNet'
        'Dropout_LeNet'
        'AlexNet_CIFAR10'
        'VGG11_CIFAR10'
    '''
    batch_size = 256
    epochs = 200
    learning_rate = 0.01
    momentum = 0.9

    weight_decay = 0.0001
    dropout_rate = 0.5

    factor = 0.5
    patience = 5
    threshold = 1e-4

    es_patience = 15 # 衰减器的耐心不要大于早停耐心