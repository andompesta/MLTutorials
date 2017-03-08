import tensorflow as tf
import numpy as np
import tempfile

contexts = [[1, 2, 3], [1, 2, 2], [4, 5, 1, 3, 2, 6], [1, 2, 3, 1, 6]]
utterances = [[1, 1, 1], [2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4]]
distractors = [
    [[11, 11], [11, 11]],
    [[22, 22], [22, 22]],
    [[33, 33], [33, 33, 33]],
    [[44, 44, 44, 44], [44, 44, 44]]
]

NUM_EXAMPLE_IN_COMPLETE_EXAMPLE = 3

####### WRITING EXAMPLE #######
def write_sequence_file(file_name, make_example_fun, path='./MY_data',):
    '''
    :param file_name: output file name
    :param make_example_fun: function used to make the example
    :param path: path to the dir where to save the files
    '''
    with tf.python_io.TFRecordWriter(path + '/' + file_name) as writer:
        print("Creating TFRecords file at {}...".format(path + '/' + file_name))
        for idx, (context, utterance) in enumerate(zip(contexts, utterances)):
            example = make_example_fun(context, utterance, 1)
            writer.write(example.SerializeToString())
            for distractor in distractors[idx]:
                example = make_example_fun(context, distractor, 0)
                writer.write(example.SerializeToString())

        writer.close()
    print("Wrote to {}".format(path + '/' + file_name))

def make_example(context, utterance, label):
    # The object we return
    example = tf.train.SequenceExample()
    # A non-sequential feature of our example
    example.context.feature["context_len"].int64_list.value.append(len(context))
    example.context.feature["utterance_len"].int64_list.value.append(len(utterance))
    example.context.feature["label"].int64_list.value.append(label)

    fl_context = example.feature_lists.feature_list["context"]
    fl_utterance = example.feature_lists.feature_list["utterance"]
    for c_token in context:
        fl_context.feature.add().int64_list.value.append(c_token)
    for u_token in utterance:
        fl_utterance.feature.add().int64_list.value.append(u_token)

    return example


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
                                                                           'context_len': tf.FixedLenFeature([], dtype=tf.int64),
                                                                           'utterance_len': tf.FixedLenFeature([], dtype=tf.int64),
                                                                           'label': tf.FixedLenFeature([], dtype=tf.int64),
                                                                       },
                                                                       sequence_features={
                                                                           "context": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                                           "utterance": tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                                       })

    return context_parsed, sequence_parsed

if __name__ == '__main__':
    write_sequence_file('train.tfrecords', make_example)
    batch_size = 2
    num_eval = 2
    # Parse the example
    feature_map_batches = read_input('train.tfrecords', batch_size * NUM_EXAMPLE_IN_COMPLETE_EXAMPLE, num_eval)

    res = tf.contrib.learn.run_n(feature_map_batches, n=(num_eval*batch_size), feed_dict=None)
    for batch in res:
        print('----BATCH----')
        for idx_example in range(0, batch_size * NUM_EXAMPLE_IN_COMPLETE_EXAMPLE, NUM_EXAMPLE_IN_COMPLETE_EXAMPLE):
            print('\t--example_%d--\t' % idx_example)
            for idx_utterance in range(NUM_EXAMPLE_IN_COMPLETE_EXAMPLE):
                for key in batch.keys():
                    print(key, batch[key][idx_example + idx_utterance])



