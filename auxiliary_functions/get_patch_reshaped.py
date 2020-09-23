def get_patch_reshaped(patches, patch_shape):
    plines = patches.shape[0]
    pcols = patches.shape[1]
    if len(patch_shape) == 3:
        patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
        patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
    if len(patch_shape) == 2:
        patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1])
        patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1])
    return patches_reshaped
