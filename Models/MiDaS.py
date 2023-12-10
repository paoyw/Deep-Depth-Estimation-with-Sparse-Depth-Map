import torch


def get_model(model_type='MiDaS_small'):
    model = torch.hub.load('intel-isl/MiDaS', model_type)
    return model

def get_transform(model_type='MiDaS_small'):
    if model_type == 'MiDaS_small':
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        return transforms.small_transform