from itertools import combinations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing
K.set_floatx('float32')

from fame.data_generation import model_inputs

class DipoleModel():
    def __init__(
        self, num_jets, permutations, X, Y, recoil_factors, dipoles, pred_scale, coef_scale, J
    ):
        self.num_jets = num_jets
        self.permutations = permutations
        self.X = X[:, 2:]
        self.Y = Y
        self.recoil_factors = recoil_factors
        self.dipoles = dipoles
        self.pred_scale = pred_scale
        self.coef_scale = coef_scale
        self.J = J
    
    def build_model(self):
        '''Builds dipole model with custom loss function.'''

        num_dipoles = len(self.permutations)
        if self.num_jets > 3:
            self.phis = True
        else:
            self.phis = False

        if self.phis:
            gluon_combinations = list(combinations(range(3, self.num_jets+1), 2))
            num_dipoles = len(self.permutations)+len(gluon_combinations)*2

        # set shapes for inputs
        inputs_raw = keras.Input(shape=(self.num_jets, 4,))
        ys = keras.Input(shape=(len(self.permutations),))
        targets = keras.Input(shape=(1,))
        dipoles = keras.Input(shape=(num_dipoles,))

        inputs = [inputs_raw, ys, dipoles, targets]

        # standardise inputs
        x = self.x_scaler(inputs_raw)
        ys = self.recoil_scaler(ys)

        # get inputs into correct shape
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, ys])

        # densely connected NN
        x = layers.Dense(64, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(128, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(256, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(512, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(768, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(386, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(128, activation="tanh", kernel_initializer="glorot_uniform")(x)
        x = layers.Dense(64, activation="tanh", kernel_initializer="glorot_uniform")(x)
        outputs = layers.Dense(num_dipoles)(x)
        outputs = self.coef_scale * tf.math.sinh(outputs)

        # add custom loss to model and metrics for monitoring whilst training
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"dipole_{self.num_jets}_jets")
        model.add_loss(self.custom_loss(targets, outputs, dipoles))
        model.add_metric(self.custom_MSE(targets, outputs, dipoles), name='mse')
        model.add_metric(self.factorisation_penalty(outputs, dipoles), name='f_pen')
        self.model = model
        
    def factorisation_penalty(self, coefs, dipoles):
        '''Penalise non-sparse representation of matrix element.'''
        di = dipoles**-2
        norm = tf.reduce_sum(di, 1)
        L_pen = tf.reduce_sum(tf.multiply(tf.math.abs(coefs*dipoles), di), 1) / norm
        return self.J*L_pen

    def custom_MSE(self, y_true, y_preds, dipoles):
        '''Mean-squared error using outputs from neural network and dipoles.'''
        y_pred = tf.reduce_sum(tf.multiply(y_preds, dipoles), 1)
        # rescale predictions the same way we did for targets
        y_pred = self.y_scaler(tf.math.asinh(y_pred / self.pred_scale))
        mse = keras.losses.mean_squared_error(y_true, y_pred)
        return mse

    def custom_loss(self, y_true, y_preds, dipoles):
        '''Total loss function is the sum of MSE + factorisation penalty.'''
        mse = self.custom_MSE(y_true, y_preds, dipoles)
        f_pen = self.factorisation_penalty(y_preds, dipoles)
        return mse + f_pen

    def preprocess_inputs(self, batch_size=2**16):
        '''Preprocess input by standardising and fit scalers.'''
        self.x_scaler = preprocessing.Normalization()
        # we only use the outgoing jets as input
        self.x_scaler.adapt(self.X, batch_size=batch_size)

        self.recoil_scaler = preprocessing.Normalization()
        # log recoil factors as they can be very small
        self.recoil_scaler.adapt(tf.math.log(self.recoil_factors), batch_size=batch_size)

        self.y_scaler = preprocessing.Normalization()
        # arcsinh labels acts like log but still works if argument is negative
        scaled = tf.math.asinh(self.Y / self.pred_scale)
        self.y_scaler.adapt(scaled, batch_size=batch_size)
        # standardise targets
        self.Y = self.y_scaler(scaled)
        
    def train_model(self, checkpoint_path, batch_size=4096, learning_rate=0.001, min_delta=1E-4, loss=None, reduce_lr=True, force_save=False):
        '''Wrapper function to train model with useful monitoring tools and saving models.'''

        # since model has custom loss added can pass loss=None here
        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

        # training termination critera, min_delta is tuned such that models don't train forever
        es = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=40,
            verbose=1,
            min_delta=min_delta,
            restore_best_weights=True
        )

        callbacks = [es]
        # learning rate reduction helps convergence
        if reduce_lr:
            lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.7,
                patience=20,
                verbose=1,
                cooldown=1,
                min_delta=0.1*min_delta,
                min_lr=1E-6
            )
            callbacks.append(lr)
        # provide checkpoint_path to checkpoint model and to save models when finished training
        if checkpoint_path is not None:
            print(f"Checkpointing model in {checkpoint_path}/...")
            if force_save:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                checkpoint_path = f"{checkpoint_path}/"
            else:
                if not os.path.exists(checkpoint_path):
                    print(f"Checkpoint path: {checkpoint_path} doesn't exist.")

            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_freq='epoch'
            )

            # output loss at every epoch to training.log file for analysis
            csv_logger = keras.callbacks.CSVLogger(
                checkpoint_path + 'training.log'
            )

            # profile training for analysis of memory/speed bottlenecks
            # also useful for evaluating training/validation loss
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=checkpoint_path + 'tensorboard_logs/',
                update_freq='epoch',
                profile_batch='100,200'
            )

            callbacks.extend([checkpoint, csv_logger, tensorboard])

        # training and saving model
        try:
            history = self.model.fit(
                [self.X, tf.math.log(self.recoil_factors), self.dipoles, self.Y],
                self.Y,
                batch_size=batch_size,
                epochs=10000,
                validation_split=0.2,
                callbacks=callbacks,
                shuffle=True,
                verbose=1
            )
            if checkpoint_path is not None:
                self.model.save_weights(checkpoint_path + 'model_weights.h5')
                print(f"Model weights saved to {checkpoint_path}model_weights.h5")
                self.model.save(checkpoint_path + "model_data")
                print(f"Model saved to {checkpoint_path}model_data")

        # still save model even if ctrl+c
        except KeyboardInterrupt:
            print("\n")
            print("Interrupting training...")
            if checkpoint_path is not None:
                self.model.save_weights(checkpoint_path + 'model_weights.h5')
                print(f"Model weights saved to {checkpoint_path}model_weights.h5")
                self.model.save(checkpoint_path + "model_data")
                print(f"Model saved to {checkpoint_path}model_data")
                weights = self.model.get_weights()
            else:
                weights = self.model.get_weights()
            return weights
        return history
    
    def dipole_network_predictor(self, model, inputs, momenta, batch_size=2**16):
        if self.phis:
            phi_terms = model_inputs.calculate_cs_phis(momenta, self.num_jets, cast=True)
            if momenta.dtype != np.float32:
                momenta = momenta.astype(np.float32)
            dipoles, ys = inputs.calculate_inputs(momenta, [*phi_terms])
        else:
            if momenta.dtype != np.float32:
                momenta = momenta.astype(np.float32)
            dipoles, ys = inputs.calculate_inputs(momenta)
            
        coefs = model.predict(
            [momenta[:, 2:], np.log(ys), np.zeros_like(dipoles.numpy()), np.zeros(len(momenta))],
            verbose=1,
            batch_size=batch_size
        )
        prediction = tf.reduce_sum(tf.multiply(coefs, dipoles), axis=1)
        return coefs, dipoles, prediction.numpy()
