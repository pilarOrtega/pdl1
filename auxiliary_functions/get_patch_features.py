import numpy

def get_patch_features(patch, features):
    features = [f[1] for f in features if f[0] in patch]
    return numpy.asarray(features)
