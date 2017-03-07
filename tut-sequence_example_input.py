import tensorflow as tf
import numpy as np
import tempfile

sequences = [[1, 2, 3], [1, 2, 2], [4, 5, 1, 3, 2, 6], [1, 2, 3, 1, 6]]
labels = [1, 0, 0, 1]



####### WRITING EXAMPLE #######
def write_sequence_file(file_name, make_example_fun, path='./MY_data',):
    '''
    :param file_name: output file name
    :param make_example_fun: function used to make the example
    :param path: path to the dir where to save the files
    '''
    with tf.python_io.TFRecordWriter(path + '/' + file_name) as writer:
        print("Creating TFRecords file at {}...".format(path + '/' + file_name))
        for sequence, label in zip(sequences, labels):
            example = make_example_fun(sequence, label)
            writer.write(example.SerializeToString())
        writer.close()
    print("Wrote to {}".format(path + '/' + file_name))

def make_example(sequence, label):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    ex.context.feature['label'].int64_list.value.append(label)

    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    for token in sequence:
        fl_tokens.feature.add().int64_list.value.append(token)
    return ex


####### READING EXAMPLE #######
def read_input(file_name, batch_size, num_epochs, path='./MY_data'):
    with tf.name_scope('input'):
        file_name_queue = tf.train.string_input_producer([path + '/' + file_name], num_epochs=num_epochs)

        context_parsed, sequence_parsed = read_and_decode(file_name_queue)      # read each example

        feature_map = dict(context_parsed, **sequence_parsed)                   # construct a feature map dictionary
        feature_map_batch = tf.train.batch(    # BATCH THE DATA
            tensors=feature_map,
            batch_size=batch_size,
            dynamic_pad=True)                                # dimanic size of the batch
    return feature_map_batch

def read_and_decode(file_name_queue):
    '''
    Decode a single sequence example
    :param file_name_queue: queue of the files to read
    :return: tensor containing the contex_feature and sequence_fature of a single example
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                                context_features={
                                                    'length': tf.FixedLenFeature([], dtype=tf.int64),
                                                    'label': tf.FixedLenFeature([], dtype=tf.int64)},
                                                sequence_features={
                                                    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                })

    return context_parsed, sequence_parsed

if __name__ == '__main__':
    write_sequence_file('train.tfrecords', make_example)

    # Parse the example
    feature_map_batches = read_input('train.tfrecords', 2, 2)

    res = tf.contrib.learn.run_n(feature_map_batches, n=4, feed_dict=None)
    print(res)



