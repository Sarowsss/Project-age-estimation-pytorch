import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet", task="classification"):
    if task == "classification" or task == "ldl":
        model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        dim_feats = model.last_linear.in_features
        model.last_linear = nn.Linear(dim_feats, num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        return model
    else :
        model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        dim_feats = model.last_linear.in_features
        model.last_linear = nn.Linear(dim_feats, 1)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        return model
    

def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
