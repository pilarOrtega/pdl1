# coding: utf8
"""
A model to perform domain adaption.

It permits to use a trained neural network and
adapt its representation to your data in an weakly supervised way.
"""
# coding: utf8
import os
import shutil
import numpy
import pickle
from skimage.io import imread
import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
import keras.applications.xception as xce
from sklearn.cluster import KMeans
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tqdm import tqdm
from keras.optimizers import Adam
from ..description import networks
from keras.layers import Lambda, Dropout
from keras.constraints import unit_norm
from keras.callbacks import Callback
from itertools import combinations
from ..data.datagen import SingleClassBatchGeneratorFromFolder


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class NoClassFoundError(Error):
    """
    Raise when no class is found in a datafolder.

    *********************************************
    """

    pass


class NotEnoughSamples(Error):
    """
    Raise when not enough samples are found for fitting.

    *********************************************
    """

    pass


########################################################################
#
# FUNCTIONS AND CLASSES
#
########################################################################

def create_monitor(datafolder, batch_size, steps, info_dir, every_n=100):
    """
    Create a keras callback class to monitor cosine similarity.

    ***********************************************************
    """
    class Monitor(Callback):
        def on_train_begin(self, logs={}):
            self.info_dir = info_dir
            self.datafolder = datafolder
            self.steps = steps
            self.batch_size = batch_size
            self.inter_cosine_dist = []
            self.intra_cosine_dist = []
            self.every_n = every_n
            self.classfolders = {}
            self.imlists = {}
            for name in os.listdir(self.datafolder):
                classdir = os.path.join(self.datafolder, name)
                if os.path.isdir(classdir):
                    self.classfolders[name] = classdir
                    imlist = []
                    for imname in os.listdir(classdir):
                        if imname.endswith(".png"):
                            filename = os.path.join(classdir, imname)
                            imlist.append(filename)
                    self.imlists[name] = imlist

        def on_batch_end(self, batch, logs={}):
            if batch % self.every_n == 0:
                # create a new model from the global one
                eval_model = keras.Model(inputs=self.model.input,
                                         outputs=self.model.get_layer("features").output)
                cosine_inter = {}
                cosine_intra = {}
                centroids = {}
                for classname, classfolder in self.classfolders.items():
                    gen = SingleClassBatchGeneratorFromFolder(self.imlists[classname], self.batch_size)
                    # print("Evaluation of class: {}".format(classname))
                    preds = eval_model.predict_generator(gen, steps=steps)
                    n = numpy.linalg.norm(preds, axis=1)
                    preds /= n[:, numpy.newaxis]
                    # compute cosine intra
                    centroids[classname] = numpy.mean(preds, axis=0)
                    meanpreds = numpy.stack([centroids[classname] for k in range(len(preds))])
                    # cosine dist to centroid
                    dotprod = (preds * meanpreds).sum(axis=1)
                    cosine_intra[classname] = (1. - dotprod).mean(axis=0)

                for classid1, classid2 in combinations(self.classfolders.keys(), 2):
                    if classid1 < classid2:
                        k = "{}-{}".format(classid1,
                                           classid2)
                    else:
                        k = "{}-{}".format(classid2,
                                           classid1)
                    # print("Evaluation of tuple {}".format(k))
                    # compute cosine dist
                    # dot prod of 2 unit vectors
                    # perfect similarity is 1.
                    dotprod = (centroids[classid1] * centroids[classid2]).sum()
                    # I want distance, not similarity, so 1 - dotprod
                    cosine_inter[k] = (1. - dotprod)
                self.inter_cosine_dist.append(cosine_inter)
                self.intra_cosine_dist.append(cosine_intra)

        def on_train_end(self, logs={}):
            with open(os.path.join(self.info_dir, "inter_cosine.p"), "wb") as f:
                pickle.dump(self.inter_cosine_dist, f)
            with open(os.path.join(self.info_dir, "intra_cosine.p"), "wb") as f:
                pickle.dump(self.intra_cosine_dist, f)

    return Monitor


def load_data(datafolder, datalim):
    """
    Load image data.

    Given a folder and a max file number.
    """
    # first, get all filenames
    filenames = []
    for name in os.listdir(datafolder):
        if name[0] != '.' and os.path.splitext(name)[1] in [".png", ".tif", ".jpg"]:
            filenames.append(name)
    # shuffle the dataset
    numpy.random.shuffle(filenames)
    files = [os.path.join(datafolder, name) for name in filenames]
    files = files[0:datalim]
    data = []
    print("-" * 20)
    print("Loading data:")
    print("-" * 20)
    for f in tqdm(files):
        data.append(imread(f))
    return xce.preprocess_input(numpy.array(data))


def load_model(outdirectory, iteration):
    """
    Re-load a previously fine-tuned model.

    Given an iteration number (iteration to re-load).
    """
    # load json and create model
    with open(os.path.join(outdirectory, 'model/xception.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(outdirectory, "weights/iter_{}.h5".format(iteration)))
    print("Loaded model from disk")
    # compilation is useless if model is loaded for prediction only
    # loaded_model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=loss)
    return loaded_model


def save_model(model, weightdirectory, iteration):
    """
    Save the weights of the model in a h5 file.

    Given a model and its iteration number, save the model.
    """
    model.save_weights(os.path.join(weightdirectory, "iter_{}.h5".format(iteration)))


def color_cycle(size):
    """
    Define a color circle for the tsne plot of the classes.

    *******************************************************
    """
    cycle = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple']
    colors = []
    for k in range(size):
        colors.append(cycle[k % len(cycle)])
    return colors


def domain_adaption(datafolder,
                    outdir,
                    imsize,
                    epochs=20,
                    iterations=5,
                    n_clusters=10,
                    threshold=0.75,
                    datalim=25000,
                    batchsize=16,
                    metric_learning=True,
                    pdl1=False):
    """
    Adapt a neural network to a new kind of images.

    Usually a network previously trained on imagenet.
    See keras applications.
    """

    model_dir = os.path.join(outdir, 'model')
    weights_dir = os.path.join(outdir, 'weights')
    info_dir = os.path.join(outdir, 'info')

    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=None,
                                 data_format=K.image_data_format())

    # create directories
    ###############################
    if os.path.exists(outdir):
        # Be careful, this erase all previous work in this directory
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    os.makedirs(model_dir)
    os.makedirs(weights_dir)
    os.makedirs(info_dir)
    ##################################################

    # GPU for similarity matrix computation
    ###################################################
    center_t = tf.placeholder(tf.float32, (None, None))
    other_t = tf.placeholder(tf.float32, (None, None))
    center_t_norm = tf.nn.l2_normalize(center_t, dim=1)
    other_t_norm = tf.nn.l2_normalize(other_t, dim=1)
    similarity = tf.matmul(center_t_norm, other_t_norm,
                           transpose_a=False, transpose_b=True)
    ###########################################################

    # Gather images in a numpy array
    ########################################################
    if pdl1:
        # Adapted way to load the data according to existing folders in PDL1 projects
        data = []
        print("-" * 20)
        print("Loading data:")
        print("-" * 20)
        for im in tqdm(datafolder):
            data.append(imread(im))
        unlabeled_images = xce.preprocess_input(numpy.array(data))
    else:
        unlabeled_images = load_data(datafolder, datalim)

    ########################################################

    # Before we start, we instantiate and store xception pre-trained model
    ############################################
    base_model = xce.Xception(include_top=False,
                              weights='imagenet',
                              input_shape=(imsize, imsize, 3),
                              pooling='avg')
    json_string = base_model.to_json()
    with open(os.path.join(model_dir, 'xception.json'), "w") as text_file:
        text_file.write(json_string)
    save_model(base_model, weights_dir, 0)
    del base_model
    ##############

    # create a tensorflow session before we start
    K.clear_session()
    sess = tf.Session()

    # Main Loop
    ###################################
    for checkpoint in range(1, iterations + 1):

        # Load previous model
        previous_model = load_model(outdir, checkpoint - 1)

        # extract features
        print("-" * 20)
        print("predicting features:")
        print("-" * 20)
        features = previous_model.predict(unlabeled_images)
        features = numpy.array(features)

        # instance of k-means
        print("-" * 20)
        print("fitting k-means:")
        print("-" * 20)
        kmeans = KMeans(n_clusters=n_clusters).fit(features)

        # select best candidates for k-means centers in the dataset
        distances = kmeans.transform(features)
        center_idx = numpy.argmin(distances, axis=0)
        centers = numpy.array([features[i] for i in center_idx])

        # compute similarity matrix
        print("-" * 20)
        print("similarity matrix:")
        similarities = sess.run(similarity, {center_t: centers, other_t: features})
        print("similarity has shape: ", similarities.shape)
        print("similarity: ", similarities)
        print("-" * 20)

        # select images closest to centers
        print("-" * 20)
        print("reliability selection:")
        print("-" * 20)
        reliable_image_idx = numpy.unique(numpy.argwhere(similarities > threshold)[:, 1])
        print("checkpoint {}: # reliable images {}".format(checkpoint, len(reliable_image_idx)))
        sys.stdout.flush()
        images = numpy.array([unlabeled_images[i] for i in reliable_image_idx])
        int_labels = numpy.array([kmeans.labels_[i] for i in reliable_image_idx])
        labels = to_categorical(int_labels)

        # write a tsne visualization figure, to check if visualization improves
        print("-" * 20)
        print("TSNE figure:")
        tsne = TSNE(n_components=2)
        x2d = tsne.fit_transform(numpy.array([features[i] for i in reliable_image_idx]))
        print("TSNE shape: ", x2d.shape)
        print("TSNE: ", x2d)
        print("-" * 20)
        plt.figure(figsize=(6, 5))
        colors = color_cycle(n_clusters)
        for i, c, label in zip(list(range(n_clusters)), colors, [str(id) for id in range(n_clusters)]):
            print("current_label: ", i)
            print("shape of kmeans predictions: ", int_labels.shape)
            print("kmeans predictions: ", int_labels)
            print("points in masked predictions: ", (int_labels == i).sum())
            plt.scatter(x2d[int_labels == i, 0], x2d[int_labels == i, 1], c=c, label=label)
        plt.legend()
        plt.title("TSNE visualization, based on {} reliable images".format(len(reliable_image_idx)))
        plt.savefig(os.path.join(info_dir, "tsne_iter_{}.png".format(checkpoint - 1)))

        # Fine tune
        print("-" * 20)
        print("Fine tuning:")
        print("-" * 20)
        base_model = xce.Xception(include_top=False,
                                  weights='imagenet',
                                  input_shape=(imsize, imsize, 3),
                                  pooling='avg')

        # compute head of the classifier, for cosine learning
        renamer = Lambda(lambda t: t, name="features")
        regularizer = Dropout(0.8)
        if metric_learning:
            normalizer = Lambda(lambda t: K.l2_normalize(1000 * t, axis=-1))
            classifier = networks.CosineDense(n_clusters,
                                              use_bias=False,
                                              kernel_constraint=unit_norm(),
                                              activation="softmax")
            y = renamer(base_model.output)
            y = normalizer(y)
            y = regularizer(y)
            y = classifier(y)
        else:
            classifier = keras.layers.Dense(n_clusters, activation="softmax")
            y = renamer(base_model.output)
            y = regularizer(y)
            y = classifier(y)

        model = keras.Model(input=base_model.input, output=y)

        model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy")
        model.fit_generator(datagen.flow(images, labels, batch_size=batchsize), steps_per_epoch=len(images) / (batchsize + 1), epochs=epochs)
        save_model(base_model, weights_dir, checkpoint)


def fine_tune(datafolder,
              outdir,
              device,
              global_iteration,
              imsize=299,
              epochs=2,
              batchsize=16,
              lr=0.0001,
              metric_learning=True):
    """
    Adapt a neural network to a new kind of images.

    Usually a network previously trained on imagenet.
    See keras applications.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    model_dir = os.path.join(outdir, 'model')
    weights_dir = os.path.join(outdir, 'weights')
    info_dir = os.path.join(outdir, 'info')

    datagen = ImageDataGenerator(featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.,
                                 zoom_range=0.,
                                 channel_shift_range=0.,
                                 fill_mode='nearest',
                                 cval=0.,
                                 horizontal_flip=False,
                                 vertical_flip=False,
                                 rescale=None,
                                 data_format=K.image_data_format(),
                                 preprocessing_function=xce.preprocess_input)

    # create directories
    ###############################
    if os.path.exists(outdir):
        # Be careful, this erase all previous work in this directory
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    os.makedirs(model_dir)
    os.makedirs(weights_dir)
    os.makedirs(info_dir)

    # get information from datafolder
    #################################
    classes = []
    maxfiles = 0
    for name in os.listdir(datafolder):
        classdir = os.path.join(datafolder, name)
        if os.path.isdir(classdir):
            classes.append(name)
            images = [f for f in os.listdir(classdir) if f.endswith(".png")]
            maxfiles = max(maxfiles, len(images))
    n_clusters = len(classes)
    if n_clusters < 2:
        raise NoClassFoundError("data folder {} has no \
                                 class directories inside".format(datafolder))
    if maxfiles < 1000:
        raise NotEnoughSamples("Not enough samples found in \
                                {}".format(datafolder))

    # Fine tune
    print("-" * 20)
    print("Fine tuning:")
    print("-" * 20)
    K.clear_session()
    base_model = xce.Xception(include_top=False,
                              weights='imagenet',
                              input_shape=(imsize, imsize, 3),
                              pooling='avg')
    # compute head of the classifier, for cosine learning
    renamer = Lambda(lambda t: t, name="features")
    regularizer = Dropout(0.8)
    if metric_learning:
        normalizer = Lambda(lambda t: K.l2_normalize(1000 * t, axis=-1))
        classifier = networks.CosineDense(n_clusters,
                                          use_bias=False,
                                          kernel_constraint=unit_norm(),
                                          activation="softmax")
        y = renamer(base_model.output)
        y = normalizer(y)
        y = regularizer(y)
        y = classifier(y)
    else:
        classifier = keras.layers.Dense(n_clusters, activation="softmax")
        y = renamer(base_model.output)
        y = regularizer(y)
        y = classifier(y)

    model = keras.Model(input=base_model.input, output=y)
    # then evaluate representation in a callback
    Monitor = create_monitor(datafolder,
                             batchsize,
                             float(maxfiles) / (batchsize + 1))

    monitor = Monitor()

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")
    print("Going to fit on {} classes.".format(n_clusters))
    print("Max number of files in a class folder is: {}.".format(maxfiles))
    print("Expected steps per epoch: {}.".format(float(maxfiles) / (batchsize + 1)))
    print("Go for a coffee or something...")
    model.fit_generator(datagen.flow_from_directory(datafolder,
                                                    target_size=(imsize, imsize),
                                                    batch_size=batchsize),
                        steps_per_epoch=float(maxfiles) / (batchsize + 1),
                        epochs=epochs,
                        callbacks=[monitor])
    save_model(base_model, weights_dir, global_iteration)
