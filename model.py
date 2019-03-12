import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from utils import Timer, Log, pad_sequences

seed = 13
np.random.seed(seed)


class CnnLstmCrfModel:
    def __init__(self, model_name, embeddings, batch_size, constants):
        self.model_name = model_name
        self.embeddings = embeddings
        self.batch_size = batch_size

        self.use_w2v = constants.USE_W2V
        self.input_w2v_dim = constants.INPUT_W2V_DIM

        self.char_embedding = constants.CHAR_EMBEDDING
        self.nchars = constants.NCHARS
        self.input_char_dim = 50
        self.output_lstm_char_dims = [50]
        self.char_cnn_filters = {2: 16, 3: 32, 4: 16}
        self.char_cnn_hidden_layers = 2

        self.use_pos = constants.USE_POS
        self.pos_embedding_dim = constants.POS_EMBEDDING_DIM
        self.nposes = constants.NPOSES

        self.use_lstm = constants.USE_LSTM
        self.output_lstm_dims = constants.OUTPUT_LSTM_DIMS

        self.use_cnn = constants.USE_CNN
        self.cnn_filters = constants.CNN_FILTERS
        self.cnn_hidden_layers = constants.CNN_HIDDEN_LAYERS

        if not self.use_cnn and not self.use_lstm:
            raise AttributeError('Config CNN or LSTM and try again!')

        self.hidden_layers = constants.HIDDEN_LAYERS

        self.all_labels = constants.ALL_LABELS
        self.num_of_class = len(constants.ALL_LABELS)

        self.use_crf = constants.USE_CRF
        self.use_extra_loss = constants.USE_EXTRA_LOSS

        self.trained_models = constants.TRAINED_MODELS

    def _add_placeholders(self):
        """
        Adds placeholders to self
        """
        self.labels = tf.placeholder(name='labels', shape=[None, None], dtype=tf.int32)

        self.word_ids = tf.placeholder(name='word_ids', shape=[None, None], dtype=tf.int32)
        self.sequence_lens = tf.placeholder(name='sequence_lens', shape=[None], dtype=tf.int32)

        self.pos_ids = tf.placeholder(name='pos_ids', shape=[None, None], dtype=tf.int32)

        self.char_ids = tf.placeholder(name='char_ids', shape=[None, None, None], dtype=tf.int32)
        self.word_lengths = tf.placeholder(name="word_lengths", shape=[None, None], dtype=tf.int32)

        self.dropout_embedding = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_embedding')
        self.dropout_lstm = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_lstm')
        self.dropout_cnn = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_cnn')
        self.dropout_hidden = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_hidden')

        self.is_training = tf.placeholder(tf.bool, name='phase')

    @staticmethod
    def _deep_2d_cnn(cnn_input, embedding_dim, config, num_of_hidden_layers, dropout, max_pooling=True):
        cnn_input = tf.expand_dims(cnn_input, -1)

        with tf.variable_scope('cnn_first_layer'):
            cnn_outputs = []
            for k in config:
                with tf.variable_scope('cnn-{}'.format(k)):
                    filters = config[k]
                    height = k

                    pad_top = math.floor((k - 1) / 2)
                    pad_bottom = math.ceil((k - 1) / 2)
                    temp_input = tf.pad(cnn_input, [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]])

                    cnn_op = tf.layers.conv2d(
                        temp_input, filters=filters,
                        kernel_size=(height, embedding_dim),
                        padding='valid', name='cnn-{}'.format(k),
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                        activation=tf.nn.relu,
                    )  # batch, seq, 1, filter

                    cnn_outputs.append(cnn_op)
            cnn_output = tf.concat(cnn_outputs, axis=-1)
            cnn_output = tf.nn.dropout(cnn_output, dropout)

        for i in range(num_of_hidden_layers):
            with tf.variable_scope('cnn_hidden_layer-{}'.format(i + 1)):
                cnn_outputs = []
                for k in config:
                    with tf.variable_scope('cnn-{}'.format(k)):
                        filters = config[k]
                        height = k
                        pad_top = math.floor((k - 1) / 2)
                        pad_bottom = math.ceil((k - 1) / 2)
                        temp_input = tf.pad(cnn_output, [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]])

                        cnn_op = tf.layers.conv2d(
                            temp_input, filters=filters,
                            kernel_size=(height, 1),
                            padding='valid', name='cnn-{}'.format(k),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            activation=tf.nn.relu,
                        )

                        cnn_outputs.append(cnn_op)
                cnn_output = tf.concat(cnn_outputs, axis=-1)
                cnn_output = tf.nn.dropout(cnn_output, dropout)

        if max_pooling:
            final_cnn_output = tf.reduce_max(cnn_output, axis=[1, 2])
        else:
            final_cnn_output = tf.reduce_max(cnn_output, axis=2)

        return final_cnn_output

    @staticmethod
    def _multi_layer_bi_lstm(lstm_input, sequence_length, config, dropout, final_state_only=False):
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [LSTMCell(size) for size in config]
        )
        cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [LSTMCell(size) for size in config]
        )

        (output_fw, output_bw), final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw,
            lstm_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
        )

        if final_state_only:
            output_fw_f = final_state[0][-1][1]
            output_bw_f = final_state[1][-1][1]
            lstm_output = tf.concat([output_fw_f, output_bw_f], axis=-1)
        else:
            lstm_output = tf.concat([output_fw, output_bw], axis=-1)
        return tf.nn.dropout(lstm_output, dropout)

    @staticmethod
    def _self_embedding(input_ids, vocab_size, dimension, dropout):
        lookup_table = tf.get_variable(
            name='lut', dtype=tf.float32,
            shape=[vocab_size, dimension],
            initializer=tf.contrib.layers.xavier_initializer(),
            # regularizer=tf.contrib.layers.l2_regularizer(1e-4),
        )
        embeddings = tf.nn.embedding_lookup(lookup_table, input_ids, name='embedding')
        return tf.nn.dropout(embeddings, dropout)

    @staticmethod
    def _mlp_project(mlp_input, input_dim, config, num_of_class, dropout, with_time_step=False):
        if with_time_step:
            nsteps = tf.shape(mlp_input)[1]
            output = tf.reshape(mlp_input, [-1, input_dim])
        else:
            nsteps = 0
            output = mlp_input

        for i, v in enumerate(config, start=1):
            output = tf.layers.dense(
                inputs=output, units=v, name='hidden_{}'.format(i),
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                activation=tf.nn.tanh,
            )
            output = tf.nn.dropout(output, dropout)

        if num_of_class != 0:
            output = tf.layers.dense(
                inputs=output, units=num_of_class, name='final_dense',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
            )

        if with_time_step:
            if num_of_class != 0:
                return tf.reshape(output, [-1, nsteps, num_of_class])
            else:
                return tf.reshape(output, [-1, nsteps, config[-1]])
        else:
            return output

    def _add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """
        embeddings = []
        self.embedding_dim = 0

        if self.use_w2v:
            with tf.variable_scope('word_embedding'):
                lut = tf.Variable(self.embeddings, name='lut', dtype=tf.float32, trainable=False)
                word_embeddings = tf.nn.embedding_lookup(
                    lut, self.word_ids,
                    name='word_embeddings'
                )
                word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_embedding)

                embeddings.append(word_embeddings)
                self.embedding_dim += self.input_w2v_dim

        if self.char_embedding.lower() != '0':
            with tf.variable_scope('char_embedding'):
                # batch, max_sent_length, seq, 50
                char_embeddings = self._self_embedding(
                    input_ids=self.char_ids,
                    vocab_size=self.nchars, dimension=self.input_char_dim,
                    dropout=self.dropout_embedding
                )

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.input_char_dim])
                # batch*max_sent_length, seq, 50
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])

                if self.char_embedding.lower() == 'lstm':
                    with tf.variable_scope('bi_lstm_char'):
                        # batch*max_sent_length, 50*2
                        lstm_output_char = self._multi_layer_bi_lstm(
                            lstm_input=char_embeddings, sequence_length=word_lengths,
                            config=self.output_lstm_char_dims, dropout=self.dropout_cnn,
                            final_state_only=True
                        )
                        # batch, max_sent_length, 50*2
                        lstm_output_char = tf.reshape(lstm_output_char, shape=[-1, s[1], 2 * self.output_lstm_char_dims[-1]])
                    embeddings.append(lstm_output_char)
                    self.embedding_dim += 2 * self.output_lstm_char_dims[-1]
                elif self.char_embedding.lower() == 'cnn':
                    with tf.variable_scope('cnn_char'):
                        final_cnn_output = self._deep_2d_cnn(
                            cnn_input=char_embeddings, embedding_dim=self.input_char_dim,
                            config=self.char_cnn_filters, num_of_hidden_layers=self.char_cnn_hidden_layers,
                            dropout=self.dropout_cnn
                        )
                        output_char_embeddings_dim = sum(self.char_cnn_filters.values())
                        final_cnn_output = tf.reshape(final_cnn_output, shape=[-1, s[1], output_char_embeddings_dim])
                    embeddings.append(final_cnn_output)
                    self.embedding_dim += output_char_embeddings_dim

        if self.use_pos:
            with tf.variable_scope('pos_embedding'):
                pos_embeddings = self._self_embedding(
                    input_ids=self.pos_ids,
                    vocab_size=self.nposes, dimension=self.pos_embedding_dim,
                    dropout=self.dropout_embedding
                )

                embeddings.append(pos_embeddings)
                self.embedding_dim += self.pos_embedding_dim

        with tf.variable_scope('final_embedding'):
            if len(embeddings) == 0:
                raise AttributeError('Empty embedding configs')
            elif len(embeddings) == 1:
                self.embeddings = embeddings[-1]
            else:
                self.embeddings = tf.concat(embeddings, axis=-1)

    def _add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope('cnn'):
            if self.use_cnn:
                total_cnn_filters = sum(self.cnn_filters.values())
                cnn_output = self._deep_2d_cnn(
                    self.embeddings, self.embedding_dim,
                    self.cnn_filters, self.cnn_hidden_layers,
                    self.dropout_cnn, max_pooling=False,
                )
                cnn_output_dim = total_cnn_filters
            else:
                cnn_output = None
                cnn_output_dim = 0

        with tf.variable_scope('bi_lstm'):
            if self.use_lstm:
                lstm_output = self._multi_layer_bi_lstm(
                    self.embeddings, self.sequence_lens,
                    self.output_lstm_dims, self.dropout_lstm,
                    final_state_only=False,
                )
                lstm_output_dim = 2 * self.output_lstm_dims[-1]
            else:
                lstm_output = None
                lstm_output_dim = 0

        with tf.variable_scope('mlp_proj'):
            inputs = []
            if self.use_cnn:
                with tf.variable_scope('cnn_logit'):
                    self.cnn_logits = self._mlp_project(
                        mlp_input=cnn_output, input_dim=cnn_output_dim, config=[],
                        num_of_class=self.num_of_class, dropout=self.dropout_hidden, with_time_step=True,
                    )
                inputs.append(cnn_output)

            if self.use_lstm:
                with tf.variable_scope('lstm_logit'):
                    self.lstm_logits = self._mlp_project(
                        mlp_input=lstm_output, input_dim=lstm_output_dim, config=[],
                        num_of_class=self.num_of_class,
                        dropout=self.dropout_hidden, with_time_step=True,
                    )
                inputs.append(lstm_output)

            with tf.variable_scope('logit'):
                if len(inputs) == 0:
                    raise AttributeError('Empty features')
                elif len(inputs) == 1:
                    mlp_input = inputs[-1]
                else:
                    mlp_input = tf.concat(inputs, axis=-1)
                mlp_input_dim = cnn_output_dim+lstm_output_dim

                self.logits = self._mlp_project(
                    mlp_input=mlp_input, input_dim=mlp_input_dim, config=self.hidden_layers,
                    num_of_class=self.num_of_class, dropout=self.dropout_hidden, with_time_step=True,
                )

    def _add_loss_op(self):
        """
        Adds loss to self
        """
        with tf.variable_scope('loss_layers'):
            if self.use_crf:
                # loss
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    inputs=self.logits, tag_indices=self.labels, sequence_lengths=self.sequence_lens
                )
                self.loss = tf.reduce_mean(-log_likelihood)

                # pred
                viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                    potentials=self.logits, transition_params=transition_params, sequence_length=self.sequence_lens
                )
                self.labels_pred = viterbi_sequence
            else:
                # pred
                self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

                # loss
                mask = tf.sequence_mask(self.sequence_lens)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

            if self.use_extra_loss and (self.use_cnn and self.use_lstm):
                mask = tf.sequence_mask(self.sequence_lens)

                # lstm loss
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cnn_logits, labels=self.labels)
                losses = tf.boolean_mask(losses, mask)
                self.loss += tf.reduce_mean(losses)

                # cnn loss
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.lstm_logits, labels=self.labels)
                losses = tf.boolean_mask(losses, mask)
                self.loss += tf.reduce_mean(losses)

    def _add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope('train_step'):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def build(self):
        timer = Timer()
        timer.start('Building model...')

        self._add_placeholders()
        self._add_word_embeddings_op()

        self._add_logits_op()
        self._add_loss_op()

        self._add_train_op()
        # f = tf.summary.FileWriter('tensorboard')
        # f.add_graph(tf.get_default_graph())
        # f.close()
        # exit(0)
        timer.stop()

    def _loss(self, sess, feed_dict):
        feed_dict = feed_dict
        feed_dict[self.dropout_embedding] = 1.0
        feed_dict[self.dropout_lstm] = 1.0
        feed_dict[self.dropout_cnn] = 1.0
        feed_dict[self.dropout_hidden] = 1.0
        feed_dict[self.is_training] = False

        loss = sess.run(self.loss, feed_dict=feed_dict)

        return loss

    def _next_batch(self, data):
        """

        :param dataset.Dataset data:
        :return:
        """
        start = 0
        idx = 0
        while start < len(data.words):
            l_batch = data.labels[start:start + self.batch_size]
            labels, _ = pad_sequences(l_batch, pad_tok=0, nlevels=1)

            w_batch = data.words[start:start + self.batch_size]
            c_batch = data.chars[start:start + self.batch_size]
            pos_batch = data.poses[start:start + self.batch_size]
            word_ids, sequence_lengths = pad_sequences(w_batch, pad_tok=0, nlevels=1)
            char_ids, word_lengths = pad_sequences(c_batch, pad_tok=0, nlevels=2)
            pos_ids, _ = pad_sequences(pos_batch, pad_tok=0, nlevels=1)

            start += self.batch_size
            idx += 1
            batch_data = {
                self.sequence_lens: sequence_lengths,
                self.labels: labels,
                self.word_ids: word_ids,
                self.char_ids: char_ids,
                self.word_lengths: word_lengths,
                self.pos_ids: pos_ids,
            }
            yield batch_data

    def _train(self, epochs, early_stopping=True, patience=10, verbose=True):
        Log.verbose = verbose
        if not os.path.exists(self.trained_models):
            os.makedirs(self.trained_models)

        saver = tf.train.Saver(max_to_keep=10)
        best_loss = float('inf')
        nepoch_noimp = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for e in range(epochs):
                self.dataset_train.shuffle()

                for idx, batch_data in enumerate(self._next_batch(data=self.dataset_train)):
                    feed_dict = {
                        **batch_data,
                        self.dropout_embedding: 0.5,
                        self.dropout_lstm: 0.5,
                        self.dropout_cnn: 0.5,
                        self.dropout_hidden: 0.5,
                        self.is_training: True,
                    }

                    _, loss_train = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if idx % 5 == 0:
                        Log.log("Iter {}, Loss: {} ".format(idx, loss_train))

                Log.log("End epochs {}".format(e + 1))
                saver.save(sess, self.model_name + '_ep{}'.format(e + 1))

                if early_stopping:
                    # stop by loss
                    total_loss = []

                    for batch_data in self._next_batch(data=self.dataset_validation):
                        feed_dict = {
                            **batch_data
                        }

                        loss = self._loss(sess, feed_dict=feed_dict)
                        total_loss.append(loss)

                    val_loss = np.mean(total_loss)
                    Log.log('Val loss: {}'.format(val_loss))
                    if val_loss < best_loss:
                        saver.save(sess, self.model_name)
                        Log.log('Save the model at epoch {}'.format(e + 1))
                        best_loss = val_loss
                        nepoch_noimp = 0
                    else:
                        nepoch_noimp += 1
                        Log.log("Number of epochs with no improvement: {}".format(nepoch_noimp))
                        if nepoch_noimp >= patience:
                            print("Best loss: {}".format(best_loss))
                            break

            if not early_stopping:
                saver.save(sess, self.model_name)

    def load_data(self, train, validation):
        """
        :param dataset.Dataset train:
        :param dataset.Dataset validation:
        :return:
        """
        timer = Timer()
        timer.start("Loading data")

        self.dataset_train = train
        self.dataset_validation = validation

        print("Number of training examples:", len(self.dataset_train.labels))
        print("Number of validation examples:", len(self.dataset_validation.labels))
        timer.stop()

    def run_train(self, epochs, early_stopping=True, patience=10):
        timer = Timer()
        timer.start("Training model...")
        self._train(epochs=epochs, early_stopping=early_stopping, patience=patience)
        timer.stop()

    # test
    def predict_on_test(self, test):
        """

        :param dataset.Dataset test:
        :return:
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            Log.log("Testing model over test set")
            saver.restore(sess, self.model_name)

            y_pred = []

            for batch_data in self._next_batch(data=test):
                feed_dict = {
                    **batch_data,
                    self.dropout_embedding: 1.0,
                    self.dropout_lstm: 1.0,
                    self.dropout_cnn: 1.0,
                    self.dropout_hidden: 1.0,
                    self.is_training: False,
                }
                preds = sess.run(self.labels_pred, feed_dict=feed_dict)
                y_pred.extend(preds)

        return y_pred
