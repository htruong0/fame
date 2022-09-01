import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
K.set_floatx('float64')

from fame_pp.data_generation import model_inputs
from fame_pp.model.custom_layers import CoefSinhLayer, CoefLayer


class DipoleModel():
    def __init__(
        self,
        num_jets,
        permutations,
        X,
        Y,
        colour,
        recoil_factors,
        dipoles,
        mandelstams,
        pred_scale,
        coef_scale,
        J,
        nodes,
        activation_function="relu",
        kernel_initialiser="he_uniform"
    ):
        self.num_jets = num_jets
        self.permutations = permutations
        self.X = X
        self.Y = Y
        self.colour = colour
        self.recoil_factors = recoil_factors
        self.dipoles = dipoles
        self.mandelstams = mandelstams
        self.pred_scale = pred_scale
        self.coef_scale = coef_scale
        self.J = np.float64(J)
        self.nodes = nodes
        self.activation_function = activation_function
        self.kernel_initialiser = kernel_initialiser
        self.csl = CoefSinhLayer(self.coef_scale)
    
    def build_model(self):
        '''Builds dipole model with custom loss function.'''

        num_dipoles = self.dipoles.shape[1]
        num_colour = self.colour.shape[1]
        num_recoils = self.recoil_factors.shape[1]
        num_sij = self.mandelstams.shape[1]

        # set shapes for inputs
        inputs_raw = keras.Input(shape=(self.num_jets, 4,))
        colour = keras.Input(shape=(num_colour, 3,))
        rfs = keras.Input(shape=(num_recoils,))
        sijs = keras.Input(shape=(num_sij,))
        targets = keras.Input(shape=(1,))
        dipoles = keras.Input(shape=(num_dipoles,))

        inputs = [inputs_raw, colour, rfs, sijs, dipoles, targets]

        # standardise inputs
        x = self.x_scaler(inputs_raw)
        c = self.colour_scaler(colour)
        rfs = self.recoil_scaler(rfs)
        s = self.sij_scaler(sijs)

        # get inputs into correct shape
        x = layers.Flatten()(x)
        c = layers.Flatten()(c)
        x = layers.Concatenate()([x, c, rfs, s])
        
        act_func = self.activation_function
        initialiser = self.kernel_initialiser
        
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(num_dipoles, activation=act_func)(x)
        coef_output = CoefLayer(self.coef_scale)(x)
        outputs = [x, coef_output]

        # add custom loss to model and metrics for monitoring whilst training
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"dipole_{self.num_jets}_jets")
        model.add_loss(self.custom_loss(targets, outputs, dipoles))
        self.model = model  
    
    def build_bare_model(self):
        num_dipoles = self.dipoles.shape[1]
        num_colour = self.colour.shape[1]
        num_recoils = self.recoil_factors.shape[1]
        num_sij = self.mandelstams.shape[1]

        # set shapes for inputs
        inputs_raw = keras.Input(shape=(self.num_jets, 4,))
        colour = keras.Input(shape=(num_colour, 3,))
        rfs = keras.Input(shape=(num_recoils,))
        sijs = keras.Input(shape=(num_sij,))

        inputs = [inputs_raw, colour, rfs, sijs]

        # standardise inputs
        x = self.x_scaler(inputs_raw)
        c = self.colour_scaler(colour)
        rfs = self.recoil_scaler(rfs)
        s = self.sij_scaler(sijs)

        # get inputs into correct shape
        x = layers.Flatten()(x)
        c = layers.Flatten()(c)
        x = layers.Concatenate()([x, c, rfs, s])
        
        act_func = self.activation_function
        initialiser = self.kernel_initialiser
        
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(self.nodes, activation=act_func, kernel_initializer=initialiser)(x)
        x = layers.Dense(num_dipoles, activation=act_func)(x)
        coef_output = CoefLayer(self.coef_scale)(x)
        outputs = [x, coef_output]

        # add custom loss to model and metrics for monitoring whilst training
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"dipole_{self.num_jets}_jets")
        self.bare_model = model
    
    
    def build_base_model(self):
        inputs_raw = keras.Input(shape=(self.num_jets, 4,))
        
        x = self.x_scaler(inputs_raw)
        x = layers.Flatten()(x)
        x = layers.Dense(self.nodes, activation="relu", kernel_initializer="he_uniform")(x)
        x = layers.Dense(self.nodes, activation="relu", kernel_initializer="he_uniform")(x)
        x = layers.Dense(self.nodes, activation="relu", kernel_initializer="he_uniform")(x)
        x = layers.Dense(self.nodes, activation="relu", kernel_initializer="he_uniform")(x)
        output = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs_raw, outputs=output, name="base_model")
        self.base_model = model
    
    def factorisation_penalty(self, coefs, dipoles):
        '''Penalise non-sparse representation of matrix element.'''
        di = tf.math.pow(dipoles, -2)
        norm = tf.math.reduce_sum(di, axis=1)
        L_pen = tf.math.divide(tf.math.reduce_sum(
                    tf.math.multiply(
                        tf.math.abs(tf.math.multiply(coefs, dipoles)), di),
                    axis=1), norm)
        return tf.math.multiply(self.J, L_pen)

    def custom_MSE(self, y_true, y_preds, dipoles):
        '''Mean-squared error using outputs from neural network and dipoles.'''
        y_preds = self.csl(y_preds[0])
        y_pred = tf.math.reduce_sum(tf.math.multiply(y_preds, dipoles), axis=1)
        # rescale predictions the same way we did for targets
        y_pred = self.y_scaler(tf.math.asinh(y_pred / self.pred_scale)[:, None])
        mse = keras.losses.mean_squared_error(y_true, y_pred)
        return mse

    def custom_loss(self, y_true, y_preds, dipoles):
        '''Total loss function is the sum of MSE + factorisation penalty.'''
        mse = self.custom_MSE(y_true, y_preds, dipoles)
#         f_pen = self.factorisation_penalty(y_preds, dipoles)
#         return tf.math.add(mse, f_pen)
        return mse

    def preprocess_inputs(self, batch_size=2**16):
        '''Preprocess input by standardising and fit scalers.'''
        self.x_scaler = layers.Normalization(axis=2)
        # we only use the outgoing jets as input
        self.x_scaler.adapt(self.X, batch_size=batch_size)

        self.colour_scaler = layers.Normalization(axis=2)
        self.colour_scaler.adapt(self.colour, batch_size=batch_size)

        self.recoil_scaler = layers.Normalization(axis=1)
        # log recoil factors as they can be very small
        self.recoil_scaler.adapt(tf.math.log(self.recoil_factors), batch_size=batch_size)
        
        self.sij_scaler = layers.Normalization(axis=1)
        self.sij_scaler.adapt(tf.math.log(self.mandelstams), batch_size=batch_size)

        self.y_scaler = layers.Normalization(axis=None)
        # arcsinh labels acts like log but still works if argument is negative
        self.scaled = tf.math.asinh(self.Y / self.pred_scale)
        self.y_scaler.adapt(self.scaled, batch_size=batch_size)
        # standardise targets
        self.Y = self.y_scaler(self.scaled)
                
    def train_model(
        self,
        checkpoint_path,
        epochs=10000,
        batch_size=4096,
        learning_rate=0.001,
        min_delta=1E-5,
        loss=None,
        reduce_lr=True,
        force_save=False,
        mode="dipole"
    ):
        '''Wrapper function to train model with useful monitoring tools and saving models.'''

        # since model has custom loss added can pass loss=None here
        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
        if hasattr(self, "base_model"):
            self.base_model.compile(
                loss=loss,
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
            )

        # training termination critera, min_delta is tuned such that models don't train forever
        es = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=60,
            verbose=1,
            min_delta=min_delta,
            restore_best_weights=True
        )
        term = keras.callbacks.TerminateOnNaN()
        callbacks = [es, term]
        # learning rate reduction helps convergence
        if reduce_lr:
            lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.7,
                patience=30,
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
                checkpoint_path / 'training.log'
            )

            # profile training for analysis of memory/speed bottlenecks
            # also useful for evaluating training/validation loss
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=checkpoint_path / 'tensorboard_logs/',
                update_freq='epoch',
                profile_batch='100,200'
            )

            callbacks.extend([checkpoint, csv_logger, tensorboard])

        # training and saving model
        try:
            if mode == 'dipole':
                inputs = [
                    self.X,
                    self.colour,
                    tf.math.log(self.recoil_factors),
                    tf.math.log(self.mandelstams),
                    self.dipoles,
                    self.Y
                ]
                model = self.model
            elif mode == 'base':
                inputs = self.X
                model = self.base_model
            print(mode)
            history = model.fit(
                inputs,
                self.Y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=callbacks,
                shuffle=True,
                verbose=1
            )
            if checkpoint_path is not None:
                model.save_weights(checkpoint_path / 'model_weights.h5')
                print(f"Model weights saved to {checkpoint_path}/model_weights.h5")
                model.save(checkpoint_path / "model_data")
                model.save(checkpoint_path / "model_data.h5")
                print(f"Model saved to {checkpoint_path}/model_data")

        # still save model even if ctrl+c
        except KeyboardInterrupt:
            print("\n")
            print("Interrupting training...")
            if checkpoint_path is not None:
                model.save_weights(checkpoint_path / 'model_weights.h5')
                print(f"Model weights saved to {checkpoint_path}/model_weights.h5")
                model.save(checkpoint_path / "model_data")
                model.save(checkpoint_path / "model_data.h5")
                print(f"Model saved to {checkpoint_path}/model_data")
                weights = model.get_weights()
            else:
                weights = model.get_weights()
            return weights
        return history


    def dipole_network_predictor(self, model, inputs, momenta, colour, CS, model_type="bare", batch_size=2**16):
        phi_terms = model_inputs.calculate_cs_phis(momenta, CS)
        dipoles, rfs, sijs = inputs.calculate_inputs(momenta, to_concat=phi_terms)
        
        if model_type == "bare":
            coefs, coef_scales = model.predict(
                [momenta, colour, tf.math.log(rfs), tf.math.log(sijs)],
                verbose=1,
                batch_size=batch_size
            )
        else:
            coefs, coef_scales = model.predict(
                [momenta, colour, tf.math.log(rfs), tf.math.log(sijs), np.zeros_like(dipoles), np.zeros(len(momenta))],
                verbose=1,
                batch_size=batch_size
            )
        coefs = coef_scales*tf.math.sinh(coefs)
        prediction = tf.reduce_sum(tf.multiply(coefs, dipoles), axis=1)
        return coefs, rfs, sijs, dipoles, prediction.numpy()

