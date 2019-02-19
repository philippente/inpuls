""" This file defines policy optimization for a tensorflow policy. """
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from gps.algorithm.policy_opt.policy_opt import PolicyOpt


class Self_Regulating_Policy(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """

    def __init__(self, hyperparams, dX, dU):
        PolicyOpt.__init__(self, hyperparams, dX, dU)
        self.dX = dX
        self.dU = dU

        tf.set_random_seed(self._hyperparams['random_seed'])
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.epochs = self._hyperparams['epochs']
        self.batch_size = self._hyperparams['batch_size']
        self.weight_decay = self._hyperparams['weight_decay']
        self.N_hidden = self._hyperparams['N_hidden']
        self.dropout_rate = self._hyperparams['dropout_rate']

        self.graph = tf.Graph()  # Encapsulate model in own graph
        with self.graph.as_default():
            self.init_network()
            self.init_loss_function()
            self.init_solver()

            # Create session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # Prevent GPS from hogging all memory
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

        self.policy = self  # Act method is contained in this class
        self.scaler = None

    def init_network(self):
        self.state_in = tf.placeholder(tf.float32, (None, self.dX))
        self.action_in = tf.placeholder(tf.float32, (None, self.dU))
        self.precision_in = tf.placeholder(tf.float32, (None, self.dU, self.dU))
        self.is_training = tf.placeholder(tf.bool, ())

        with tf.variable_scope('action_estimator'), arg_scope(
            [layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=layers.l2_regularizer(scale=self.weight_decay)
        ):
            h = layers.fully_connected(self.state_in, self.N_hidden)
            h = layers.dropout(h, keep_prob=1 - self.dropout_rate, is_training=self.is_training)
            h = layers.fully_connected(h, self.N_hidden)
            h = layers.dropout(h, keep_prob=1 - self.dropout_rate, is_training=self.is_training)
            h = layers.fully_connected(h, self.N_hidden)
            h = layers.dropout(h, keep_prob=1 - self.dropout_rate, is_training=self.is_training)
            self.action_estimation = layers.fully_connected(h, self.dU, activation_fn=None)

        with tf.variable_scope('stabilizer_estimator'), arg_scope(
            [layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=layers.l2_regularizer(scale=self.weight_decay)
        ):
            i1 = self.state_in
            i2 = self.action_estimation
            #i1 = tf.nn.tanh(self.state_in)
            #i2 = tf.nn.tanh(self.action_estimation)
            #i1 = layers.fully_connected(self.state_in, 8, activation_fn=tf.nn.tanh)
            #i2 = layers.fully_connected(self.action_estimation, 8, activation_fn=tf.nn.tanh)
            h = layers.fully_connected(tf.concat([i1, i2], axis=-1), self.N_hidden)
            h = layers.fully_connected(h, self.N_hidden)
            h = layers.fully_connected(h, self.N_hidden)
            self.stabilizer_estimation = tf.reshape(
                layers.fully_connected(h, self.dX * self.dU, activation_fn=None), (-1, self.dU, self.dX)
            )
        self.action_out = tf.einsum('inm,im->in', self.stabilizer_estimation, self.state_in) + self.action_estimation

    def init_loss_function(self):
        # KL divergence loss
        #  loss_kl = 1/2 delta_action^T * prc * delta_action
        delta_action_estimation = self.action_in - self.action_estimation
        self.loss_kl_action_estimation = tf.reduce_mean(
            tf.einsum('in,inm,im->i', delta_action_estimation, self.precision_in, delta_action_estimation)
        ) / 2

        delta_action = self.action_in - self.action_out
        self.loss_kl_action = tf.reduce_mean(
            tf.einsum('in,inm,im->i', delta_action, self.precision_in, delta_action)
        ) / 2

        # Regularization loss
        self.loss_reg_action = tf.losses.get_regularization_loss(scope='action_estimator')
        self.loss_reg_stabilizer = tf.losses.get_regularization_loss(scope='stabilizer_estimator')

        # Total loss
        self.loss_action = self.loss_kl_action_estimation + self.loss_reg_action
        self.loss_stabilizer = self.loss_kl_action + self.loss_reg_stabilizer

    def init_solver(self):
        # Optimizer for action estimator
        optimizer_action = tf.train.AdamOptimizer()
        self.solver_op_action = optimizer_action.minimize(
            loss=self.loss_action,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='action_estimator'),
        )
        optimizer_action_reset_op = tf.variables_initializer(optimizer_action.variables())

        # Optimizer for stabilizer estimator
        optimizer_stabilizer = tf.train.AdamOptimizer()
        self.solver_op_stabilizer = optimizer_stabilizer.minimize(
            loss=self.loss_stabilizer,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stabilizer_estimator'),
        )
        optimizer_stabilizer_reset_op = tf.variables_initializer(optimizer_stabilizer.variables())

        self.optimizer_reset_ops = [optimizer_action_reset_op, optimizer_stabilizer_reset_op]

    def update(self, X, mu, prc, _):
        """
        Trains a GPS model on the dataset
        """
        N, T, _ = X.shape

        # Reshape inputs.
        X = X.reshape((N * T, self.dX))
        mu = mu.reshape((N * T, self.dU))
        prc = prc.reshape((N * T, self.dU, self.dU))

        # Normalize X, but only compute normalization at the beginning.
        if self.scaler is None:
            self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

        # Create dataset
        with self.graph.as_default():
            dataset = tf.data.Dataset.from_tensor_slices((X, mu, prc)).shuffle(N).batch(self.batch_size)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()

        # Reset optimizer
        self.sess.run(self.optimizer_reset_ops)

        batches_per_epoch = int(N * T / self.batch_size)
        assert batches_per_epoch * self.batch_size == N * T, 'N=%d, batchsize=%d, batches_per_epoch=%d' % (
            N * T, self.batch_size, batches_per_epoch
        )

        losses = np.zeros((self.epochs, 2))
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            # Initialize dataset iterator
            self.sess.run(iterator.initializer)

            for i in range(batches_per_epoch):
                batch_X, batch_mu, batch_prc = self.sess.run(next_element)

                # Train action
                fetches = self.sess.run(
                    [self.solver_op_action, self.loss_action, self.action_estimation],
                    feed_dict={
                        self.state_in: batch_X,
                        self.action_in: batch_mu,
                        self.precision_in: batch_prc,
                        self.is_training: True,
                    }
                )
                losses[epoch, 0] += fetches[1]
                batch_action_estimation = fetches[2]

                # Train stabilizer
                fetches = self.sess.run(
                    [self.solver_op_stabilizer, self.loss_stabilizer],
                    feed_dict={
                        self.action_estimation: batch_action_estimation,
                        self.state_in: batch_X,
                        self.action_in: batch_mu,
                        self.precision_in: batch_prc,
                    }
                )
                losses[epoch, 1] += fetches[1]
            losses[epoch] /= batches_per_epoch
            pbar.set_description("Loss: %.6f/%.6f" % (losses[epoch, 0], losses[epoch, 1]))

        # Visualize training loss
        from gps.visualization import visualize_loss
        visualize_loss(
            self._data_files_dir + 'plot_gps_training-%02d' % (self.iteration_count),
            losses,
            labels=['Action estimation', 'Stabilizer estimation'],
            add_total=False
        )

        # Optimize variance.
        A = np.mean(prc, axis=0) + 2 * N * T * self._hyperparams['ent_reg'] * np.ones((self.dU, self.dU))

        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

    def act(self, x, _, t, noise):
        u = self.sess.run(
            self.action_out, feed_dict={
                self.state_in: self.scaler.transform([x]),
                self.is_training: False,
            }
        )[0]
        if noise is not None:
            if t is None:
                u += self.chol_pol_covar.T.dot(noise[0])
            else:
                u += self.chol_pol_covar.T.dot(noise[t])
        return u

    def prob(self, X):
        """
        Run policy forward.
        Args:
            X: States (N, T, dX)
        """
        N, T = X.shape[:2]

        action = self.sess.run(
            self.action_out,
            feed_dict={
                self.state_in: self.scaler.transform(X.reshape(N * T, self.dX)),
                self.is_training: False,
            }
        ).reshape((N, T, self.dU))
        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return action, pol_sigma, pol_prec, pol_det_sigma
