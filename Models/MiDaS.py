import torch


def get_model(model_type='MiDaS_small'):
    model = torch.hub.load('intel-isl/MiDaS', model_type)
    return model
