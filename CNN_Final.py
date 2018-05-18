import tensorflow as tf
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict

# Hyperparameters
global EPOCH
global MIN_TRAIN_ITER
global LEARNING_RATE
global BETA1
global BETA2
global EPSILON
global UNITS
global DROPOUT

# Global settings
global BATCH_SIZE
global CLASSES
global DATASETS_DIR
global PRINT_IMAGES
global PLOT_GRAPH
global DEBUG_RAW_TITLE_SHOW
global DEBUG_TITLE_SHOW


def initialize_queue(files, randomize=True):
    """Convert images to Tensors

    Takes a list of image files with the same file format and turns them into Tensors.

    Args:
        *files (char): List of string cointaining the path and file matching pattern.
        *classes (char): String containing the classes name.
        randomize (boolean): Organize the files in the queue in a random sequence.
            Default: True

    Returns:
        Dictionary of Tensors containing the queues with all the images which match the pattern
    """
    dataset_queue = {}

    for i in range(len(files)):
        match = tf.train.match_filenames_once(files[i])
        queue = tf.train.string_input_producer(match, shuffle=randomize)
        reader = tf.WholeFileReader()
        _, image_file = reader.read(queue)
        dataset_queue.update({'dataset' + str(i): image_file})

    return dataset_queue


def image_to_jpeg(files, labels, width=80, height=140, gray=True, normalize=True, method=0, randomize=False,
                  resize=False):
    """Configure a set of pictures

    Rescale, assign labels or turn a picture into RGB/Grayscale colormap. It also
    normalizes the pictures on a Cartesian centre.

    Args:
        *files (Tensor): List of string containing a set of pictures from a directory.
        *label (int): Integer list with a single element to label each Tensor created
            by this function.
        width (int): Image width. Default: 80
        height (int): Image height. Default: 140
        gray (boolean): True to turn a picture into grayscale. False to turn it into
            RGB colorspace. Default: True
        normalize (boolean): True to center images on the Cartesian origin. False to
            keep it as it is. Default: True
        technique (int): Select a technique to resize the images with a number mapping to:
                0 -> ResizeMethod.BILINEAR
                1 -> ResizeMethod.NEAREST_NEIGHBOR
                2 -> ResizeMethod.BICUBIC
                3 -> ResizeMethod.AREA
            Default: 0

    Returns:
        Dictionary containing images with the set options and a label dictionary
    """

    dataset_queue = initialize_queue(files, randomize=randomize)

    dataset = {}
    labelset = {}

    for i in range(len(dataset_queue)):

        if gray:
            raw_image, label = tf.image.decode_jpeg(dataset_queue['dataset' + str(i)], channels=1), [labels[i]]
        else:
            raw_image, label = tf.image.decode_jpeg(dataset_queue['dataset' + str(i)], channels=3), [labels[i]]

        if resize:
            # raw_image = tf.image.convert_image_dtype(raw_image, tf.float32)  # Esta línea está maldita: NO DESCOMENTAR
            raw_image = tf.image.resize_images(raw_image, (width, height), method=method)

        if gray:
            #raw_image = tf.image.rgb_to_grayscale(raw_image)
            raw_image = tf.reshape(raw_image, [width, height, 1])  # -1
        else:
            raw_image = tf.reshape(raw_image, [width, height, 3])  # -1

        if normalize:
            raw_image = tf.to_float(raw_image) / 256. - 0.5
        else:
            raw_image = tf.to_float(raw_image)

        dataset.update({'dataset' + str(i): raw_image})
        labelset.update({'labelset' + str(i): label})

    return dataset, labelset


def initialize_batches(dataset, labelset, batch_size=4, min_after_dequeue=10):
    # if training_percentage > 1 or training_percentage < 0:
    #    print("The training percentage must range between [0, 1]")
    #    print("Setting values by default . . .")
    #    training_percentage = 0.8

    capacity = min_after_dequeue + 3 * batch_size

    # Dictionary of batches
    sample_batches = {}
    label_batches = {}

    # List of batches
    batches = []
    labels = []

    for i in range(len(dataset)):
        sample_batch, label_batch = tf.train.shuffle_batch([dataset['dataset' + str(i)], labelset['labelset' + str(i)]],
                                                           batch_size=batch_size, capacity=capacity,
                                                           min_after_dequeue=min_after_dequeue)
        sample_batches.update({'batch' + str(i): sample_batch})
        label_batches.update({'label' + str(i): label_batch})

        batches.append(sample_batches['batch' + str(i)])
        labels.append(label_batches['label' + str(i)])
        print("=" * 40)
        print()
        print("BATCH" + str(i) + " :", batches[i])
        print("LABEL" + str(i) + " :", labels[i])
        print()
    print("=" * 40)
    print()

    # Beware here so you can extract the batches for testing set
    # for i in range(len(dataset)):

    all_samples_batch = tf.concat(values=batches, axis=0)
    all_labels_batch = tf.one_hot(indices=tf.concat(values=labels, axis=0), depth=len(labels))

    return all_samples_batch, all_labels_batch


import matplotlib as mp
import matplotlib.pyplot as plt


def show_batch_image(input_layer, input_label, debug=True, title=None, pic_title=""):
    current_batch = sess.run(tf.squeeze(input_layer), feed_dict={mode: 0})
    current_label = sess.run(tf.squeeze(input_label), feed_dict={mode: 0})

    if title != None:
        print("=" * 40)
        print()
        print(title)
        print()
        print("=" * 40)

    if debug:
        print()
        print("INPUT TENSOR :", current_batch)
        print()

    size = current_batch.shape[0]

    if debug:
        print("BATCH PICTURES SIZE :", size)

    for i in range(size):
        if debug:
            print("BATCH IMAGE #", i)

            print(current_batch[i])
            print("BATCH SHAPE :", current_batch[i].shape)

            print("ONE HOT LABEL :", current_label[i])
            print()

        plot = plt.imshow(current_batch[i])
        plt.title(pic_title + "BATCH IMAGE #" + str(i))
        plt.show()

    return None


def training_model_CNN(all_batches, mode, batch_size=4, units=100, classes=4,
                       learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, dropout=0.4):
    input_layer, labels = batch_to_model(all_batches, mode)

    o1 = tf.layers.conv2d(input_layer, filters=32, kernel_size=3, activation=tf.nn.relu)

    o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)

    o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)

    o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

    fc1 = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 4, 33 * 18 * 64]), units=units, activation=tf.nn.relu)

    fc1 = tf.cond(tf.equal(mode, 2),
                  lambda: fc1,
                  lambda: tf.layers.dropout(inputs=fc1, rate=dropout, training=tf.equal(mode, 0)))
    # fc1 = tf.layers.dropout(inputs=fc1, rate=dropout, training=tf.equal(mode, 0))

    y = tf.layers.dense(inputs=fc1, units=classes, activation=tf.nn.softmax)

    # loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(tf.squeeze(all_labels_batch), tf.float32) * tf.log(y), axis=1)) # Posible error

    loss = tf.reduce_sum(tf.square(y - tf.cast(tf.squeeze(labels), tf.float32)))

    optimizer = tf.cond(tf.equal(mode, 0),
                        lambda: tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon). \
                        minimize(loss=loss),
                        lambda: tf.no_op())

    return y, loss, optimizer, input_layer, labels


def batch_to_model(all_batches, mode):
    batch, one_hot_labels = tf.case({
        tf.equal(mode, 0): lambda: all_batches['training'],
        tf.equal(mode, 1): lambda: all_batches['validation']},
        default=lambda: all_batches['test'], exclusive=True)
    return batch, one_hot_labels


"""def initialize_validation_batches(dataset, labelset, batch_size=4, min_after_dequeue=10, classes=4):
    capacity = min_after_dequeue + 3 * batch_size

    sample_batches = {}
    label_batches = {}

    batches = []
    labels = []

    for i in range(len(dataset)):
        sample_batch, label_batch = tf.data.Dataset.from_tensor_slices(
            [dataset['dataset' + str(i)], labelset['labelset' + str(i)]],
            batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        sample_batches.update({'batch' + str(i): sample_batch})
        label_batches.update({'label' + str(i): label_batch})

    # Beware here so you can extract the batches for testing set
    for i in range(len(dataset)):
        batches.append(sample_batches['batch' + str(i)])
        labels.append(label_batches['label' + str(i)])
        print("=" * 40)
        print()
        print("BATCH" + str(i) + " :", batches[i])
        print("LABEL" + str(i) + " :", labels[i])
        print()
    print("=" * 40)
    print()

    all_samples_batch = tf.concat(values=batches, axis=0)
    all_labels_batch = tf.one_hot(indices=tf.concat(values=labels, axis=0), depth=classes)

    return all_samples_batch, all_labels_batch, sample_batches, label_batches"""

import os


def generate_datasets(dir):
    datasets = os.listdir(dir)

    all_datasets_path = {}
    num_files = defaultdict(Counter)
    path = []

    for dataset in datasets:
        classes = os.listdir(dir + '/' + dataset)
        for the_class in classes:
            path.append(dir + '/' + dataset + '/' + the_class + '/' + '*.jpg')
            num_files[dataset][the_class] = len(os.listdir(dir + '/' + dataset + '/' + the_class))
        all_datasets_path.update({dataset + '_files': path})
        path = []

    labels = [x for x in range(len(list(all_datasets_path.values())[0]))]  # labels = [0, 1, 2, 3]

    return all_datasets_path, labels, datasets, num_files


def get_batches(all_datasets_path, labels, datasets, gray=True, normalize=True, method=3, randomize=False,
                resize=False):
    all_batches = {}

    for dataset in datasets:
        # Get training batches
        the_dataset, labelset = image_to_jpeg(all_datasets_path[dataset + '_files'], labels,
                                              gray=gray, normalize=normalize, method=method,
                                              randomize=randomize, resize=resize)

        dataset_batch, one_hot_label = initialize_batches(the_dataset, labelset)

        all_batches.update({dataset: (dataset_batch, one_hot_label)})

    return all_batches


import matplotlib.pyplot as plt


def load_loss_variables(loss_graphs, epoch, cost, dataset):
    loss_graphs.update({dataset: (epoch, cost)})
    return None


def load_accuracy_variables(accuracy_graphs, epoch, accuracy, dataset):
    accuracy_graphs.update({dataset: (epoch, accuracy)})
    return None


def plot(variables, datasets, xlabel, ylabel, conf, title):
    plt.figure()
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for dataset, the_conf in zip(datasets, conf):
        plt.plot(variables[dataset][0], variables[dataset][1], the_conf, label=dataset)

    return None


def plot_loss(variables, min_epoch, min_error, datasets, conf, title='Loss vs Epoch'):
    plot(variables, datasets, 'Epoch', 'Loss', conf, title)
    plt.plot(min_epoch, min_error, conf[-1], label='Minimum error')
    plt.legend()
    plt.show()
    return None


def plot_accuracy(variables, datasets, conf, title='Accuracy vs Epoch'):
    plot(variables, datasets, 'Epoch', 'Accuracy (%)', conf, title)
    plt.legend()
    plt.show()
    return None


def start_training():
    training_epoch = []
    training_cost = []
    training_accuracy = []
    training_statistics = Counter()

    validation_epoch = []
    validation_cost = []
    validation_accuracy = []
    validation_statistics = Counter()

    min_error = 1E6
    min_epoch = 0

    loss_graphs = {}
    accuracy_graphs = {}

    #num_steps_train = sum(num_files['training'].values()) // (BATCH_SIZE * len(CLASSES))
    #num_steps_val = sum(num_files['validation'].values()) // (BATCH_SIZE * len(CLASSES))

    # Commence the training . . .
    for i in range(EPOCH):
        # Back-propagation right in here
        sess.run(optimizer, feed_dict={mode: 0})

        # Gimme the TRAINING data
        training_prediction = sess.run(tf.argmax(sess.run(y, feed_dict={mode: 0}), axis=1), feed_dict={mode: 0})
        training_loss = sess.run(loss, feed_dict={mode: 0})

        # Gimme the VALIDATION data
        validation_prediction = sess.run(tf.argmax(sess.run(y, feed_dict={mode: 1}), axis=1), feed_dict={mode: 1})
        validation_loss = sess.run(loss, feed_dict={mode: 1})

        if validation_loss < min_error:
            min_error = validation_loss
            min_epoch = i+1
            print("[+] NEW GOOD UPDATE REACHED!")
            print("EPOCH:", min_epoch, "COST:", min_error)
            print("PREDICTION:", validation_prediction)
            save_path = saver.save(sess, "./tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)

        # if i % 5 == 0:
        training_epoch.append(i+1)
        training_cost.append(training_loss)

        validation_epoch.append(i+1)
        validation_cost.append(validation_loss)

        if (i+1) % 20 == 0:
            print("-" * 22, "EPOCH:", i+1, "-" * 22)
            # print(training_stats)

            # print(current_label)
            print("TRAINING ERROR:", training_loss)
            print("TRAINING PREDICTION:", training_prediction)
            print("VALIDATION ERROR:", validation_loss)
            print("VALIDATION PREDICTION:", validation_prediction)

        if PLOT_GRAPH:
            # Gather training plot data
            # if i < num_steps_train:
            train_split_prediction = np.split(training_prediction, len(CLASSES))
            for THE_CLASS, the_label, split_prediction in zip(CLASSES, labels, train_split_prediction):
                training_statistics[THE_CLASS] += list(split_prediction).count(the_label)

            training_accuracy.append(
                float(sum(training_statistics.values()) / ((i + 1) * BATCH_SIZE * len(CLASSES)) * 100))

            # if i == num_steps_train-1:
            # training_accuracy_epoch = training_epoch.copy()

            # Gather validation plot data
            # if i < num_steps_val:
            validation_split_prediction = np.split(validation_prediction, len(CLASSES))
            for THE_CLASS, the_label, split_prediction in zip(CLASSES, labels, validation_split_prediction):
                validation_statistics[THE_CLASS] += list(split_prediction).count(the_label)

            validation_accuracy.append(
                float(sum(validation_statistics.values()) / ((i + 1) * BATCH_SIZE * len(CLASSES)) * 100))

            # if i == num_steps_val-1:
            # validation_accuracy_epoch = validation_epoch.copy()

        # Show batch pictures
        if PRINT_IMAGES:
            show_batch_image(input_layer, input_label, debug=False, title=DEBUG_RAW_TITLE_SHOW,
                             pic_title=DEBUG_TITLE_SHOW)

    load_loss_variables(loss_graphs, training_epoch, training_cost, 'training')
    load_loss_variables(loss_graphs, validation_epoch, validation_cost, 'validation')
    load_accuracy_variables(accuracy_graphs, training_epoch, training_accuracy, 'training')
    load_accuracy_variables(accuracy_graphs, validation_epoch, validation_accuracy, 'validation')

    # Plot loss graph
    if PLOT_GRAPH:
        plot_loss(loss_graphs, min_epoch, min_error, datasets=['training', 'validation'],
                  conf=['b-', 'g-', 'r-x'], title='Loss vs Epoch')
        plot_accuracy(accuracy_graphs, datasets=['validation', 'training'], conf=['r-', 'g-'],
                      title='Accuracy vs Epoch')

    return None


def start_test():
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    statistics = Counter()

    num_steps = sum(num_files['test'].values()) // (BATCH_SIZE * len(CLASSES))

    print("-" * 22, "TEST STATS", "-" * 22)

    for i in range(num_steps):
        test_stats = sess.run(y, feed_dict={mode: 2})
        # print("ITERATION = ", i, "THE TRUTH MATRIX : ", test_stats)
        test_prediction = np.split(np.array(sess.run(tf.argmax(test_stats, axis=1), feed_dict={mode: 2})),
                                   len(CLASSES))
        for THE_CLASS, the_label, split_prediction in zip(CLASSES, labels, test_prediction):
            # print("THE_CLASS = ", THE_CLASS, "THE_LABEL = ", the_label, "PREDICTION_SPLIT = ", split_prediction)

            statistics[THE_CLASS] += list(split_prediction).count(the_label)

            # print(test_prediction)
            # print(statistics)
        # print(statistics)

    # print(accuracy)

    # print("STATISTICS = ", statistics, "TOTAL FILES = ", sum(num_files[dataset].values()))

    for THE_CLASS in CLASSES:
        partial_stat = float(statistics[THE_CLASS] / num_files['test'][THE_CLASS] * 100)
        print("=" * 40)
        print("PARTIAL SUCCESS/FAILURE RATE OF %s" % THE_CLASS)
        print("=" * 40)
        print("SUCCESS: %.5f%%" % partial_stat)
        print("FAILURE: %.5f%%" % (float(100 - partial_stat)))
        print()

    total_stat = float(sum(statistics.values()) / sum(num_files['test'].values()) * 100)
    print("=" * 40)
    print("TOTAL SUCCESS/FAILURE RATE")
    print("=" * 40)
    print("SUCCESS: %.5f%%" % total_stat)
    print("FAILURE: %.5f%%" % float(100 - total_stat))
    print()


    return None

# Hyperparameters
EPOCH = 200  # 200
MIN_TRAIN_ITER = 120
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
UNITS = 100
DROPOUT = 0.3

# Global settings
BATCH_SIZE = 4
CLASSES = ['A', 'B', 'C', 'D']
DATASETS_DIR = 'dataset1/gray'
PRINT_IMAGES = False
PLOT_GRAPH = True
DEBUG_RAW_TITLE_SHOW = 'BATCHES IMAGES & LABELS SHOW (ONE HOT)'
DEBUG_TITLE_SHOW = 'TRAINING '

# Set up batches from each dataset
all_datasets_path, labels, datasets, num_files = generate_datasets(DATASETS_DIR)
all_batches = get_batches(all_datasets_path, labels, datasets,
                          gray=True, normalize=True, method=3,
                          randomize=False, resize=False)

# Set up the model
mode = tf.placeholder(tf.int32)  # Set the mode of the CNN model
y, loss, optimizer, input_layer, input_label = training_model_CNN(all_batches, mode=mode, batch_size=BATCH_SIZE,
                                                                  units=UNITS, classes=4, learning_rate=LEARNING_RATE,
                                                                  beta1=BETA1, beta2=BETA2, epsilon=EPSILON,
                                                                  dropout=DROPOUT)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_training()

    start_test()

    coord.request_stop()
    coord.join(threads)
