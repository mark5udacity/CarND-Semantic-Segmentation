import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    helper.maybe_download_pretrained_vgg(vgg_path)

    if not tf.saved_model.loader.maybe_saved_model_directory(vgg_path):
        warnings.warn("There doesn't appear to be a saved model found in path: '{}'".format(vgg_path))

    #metaGraphDef = # This is returned by .load, is a protobuf of everything, but need to go through graph to get stuff properly
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input, keep, layer_3, layer_4, layer_7

tests.test_load_vgg(load_vgg, tf)


REGULARIZER_SCALE = 1e-3

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Encoder, based on FCN-8
    conv_1x1_layer_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                padding = 'same',
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    # Decoder, based on FCN-8
    output_decoder = tf.layers.conv2d_transpose(conv_1x1_layer_7, num_classes, 4, (2, 2),
                                       padding = 'same',
                                       kernel_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    # Good example how to get sizes, add [1:3] to get y-dim
    tf.Print(output_decoder, [tf.shape(output_decoder)])

    #"""  I'm not entirely sure on these skip layers, at least make it easy to remove for further debugging if needed

    # Apply scaling, per forum post.
    # https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    # Skip Connections, based on FCN-8
    conv_1x1_layer_4 = tf.layers.conv2d(pool4_out_scaled, num_classes, 1,
                                padding = 'same',
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    skip_1 = tf.add(output_decoder, conv_1x1_layer_4) # TODO: or do we use vgg_layer7 ???
    skip_1 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2,
                                                padding='same',
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    conv_1x1_layer_3 = tf.layers.conv2d(pool3_out_scaled, num_classes, 1,
                                padding = 'same',
                                kernel_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))

    skip_2 = tf.add(skip_1, conv_1x1_layer_3)
    skip_2 = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8,
                                        padding = 'same',
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_SCALE))
    return skip_2
    #"""

    return output_decoder

tests.test_layers(layers)

REGULARIZATION_CONSTANT = 0.01  # Choose an appropriate one.
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)

    For later use in case it's needed:
    if TensorFlow’s default variable initializer for conv2d and conv2d_transpose layers (which is Glorot uniform)
     doesn’t work for you, try to change those initializers to truncated normal with a small standard deviation
      of 0.01 or even 0.001 instead. If you implemented l2-regularization correctly,
       you might not run into this problem though.
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes)) # TODO: Perhaps not needed!? But unit tests force due to shape
    #labels = tf.reshape(correct_label, (-1, num_classes)) # Let's see if this is needed?
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label)

    # Add regularization, per forum post leading to https://stackoverflow.com/q/46615623
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + REGULARIZATION_CONSTANT * sum(reg_losses)

    loss_op = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    losses = []
    steps = 0
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            steps += 1
            feed = {
                correct_label: label,
                input_image: image,
                keep_prob: .5,
                learning_rate: .001
                }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
            if steps % 5 == 0:
                print('Epoch:', epoch, 'Steps:', steps, 'Loss:', loss)

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)

    batch_size = 128
    epochs = 20

    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Path to vgg model
    vgg_path = os.path.join('.', 'vgg')

    helper.maybe_download_pretrained_vgg(vgg_path)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer_3, layer_4, layer_7 = load_vgg(sess, vgg_path)
        layer_output = layers(layer_3, layer_4, layer_7, num_classes)
        correct_label = tf.placeholder(tf.float32) # TODO: What should this be??
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
