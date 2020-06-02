def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)

def get_clusterlist(outpath, classifier, n_division):
    cluster_list = []
    image_list = glob.glob(os.path.join(outpath, '*.jpg'))
    print('Get cluster list from {}'.format(outpath))
    for im in tqdm(image_list):
        image_name = os.path.basename(im)
        image_name = image_name.split('#')[1]
        number = image_name.split('-')
        number = int(number[0])

        if classifier[number][3] == 0:
            continue

        cluster = 0
        for j in range(n_division):
            exp = n_division - j - 1
            cluster = cluster + classifier[number][j+4] * (2**exp)
        cluster_list.append((im, cluster))

    return cluster_list

def extract_complete_clusterlist(classifier, ndivision, outpath, feature_method):
    # Creamos una lista con todas las imagenes y su cluster
    clusterlist = []
    for c in classifier:
        clusterlist.extend(get_clusterlist(c[1], c[2], ndivision))

    # Creamos un set de cada cluster
    for i in range(2**ndivision):
        cluster = {x[0] for x in clusterlist if x[1] == i}
        name = 'cluster_{}_{}_{}.p'.format(feature_method, ndivision, i)
        pickle_save(cluster, outpath, name)