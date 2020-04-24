
def class_to_cluster(classifier, level, cluster):
    binary_format = '0{}b'.format(level)
    cluster_bin = format(cluster, binary_format)
    list_images_cluster = []
    for i in range(classifier.shape[0]):
        flag = True
        if classifier[i][1] == 0:
            continue
        for j in range(level):
            if (classifier[i][j+2] != int(cluster_bin[j])):
                flag = False
                break
        if flag:
            list_images_cluster.append(classifier[i][0])

    return list_images_cluster
