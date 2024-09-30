import torchio as tio

def intensity_transform():
    ## image transforms ##
    transforms_dict = {
        tio.RandomBiasField(p=1.0),
    }
    transform = tio.Compose(transforms_dict)

    return transform
