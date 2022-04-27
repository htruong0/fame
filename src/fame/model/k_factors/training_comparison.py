import sys
import os
from pathlib import Path
import numpy as np
import json
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing
K.set_floatx('float32')

from fame.data_generation import model_inputs, antennaGenerator, antenna, subantenna

def train_model(
        model,
        X,
        Y,
        checkpoint_path,
        epochs=10000,
        batch_size=512,
        learning_rate=1E-3,
        min_delta=1E-6,
        loss=None,
        reduce_lr=True,
        force_save=False
    ):
    '''Wrapper function to train model with useful monitoring tools and saving models.'''

    # since model has custom loss added can pass loss=None here
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )

    # training termination critera, min_delta is tuned such that models don't train forever
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
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
        history = model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True,
            verbose=2
        )
        if checkpoint_path is not None:
            model.save_weights(checkpoint_path + 'model_weights.h5')
            print(f"Model weights saved to {checkpoint_path}model_weights.h5")
            model.save(checkpoint_path + "model_data")
            print(f"Model saved to {checkpoint_path}model_data")

    # still save model even if ctrl+c
    except KeyboardInterrupt:
        print("\n")
        print("Interrupting training...")
        if checkpoint_path is not None:
            model.save_weights(checkpoint_path + 'model_weights.h5')
            print(f"Model weights saved to {checkpoint_path}model_weights.h5")
            model.save(checkpoint_path + "model_data")
            print(f"Model saved to {checkpoint_path}model_data")
            weights = model.get_weights()
        else:
            weights = model.get_weights()
        return weights
    return history

def build_model(n_antenna, n_maps, num_jets):   
    # leave one jet out for comparison purposes
    p_input = keras.Input(shape=(num_jets-1, 4,))
    map_input = keras.Input(shape=(n_maps,))
    antenna_input = keras.Input(shape=(n_antenna,))
    targets = keras.Input(shape=(1,))
    born = keras.Input(shape=(1,))
    
    inputs = [p_input, map_input, antenna_input, targets, born]

    mom_scaler = layers.Normalization(axis=1)
    mom_scaler.adapt(X_train[:, 3:], batch_size=2**17)

    map_scaler = layers.Normalization(axis=1)
    map_scaler.adapt(transformed_map_variables, batch_size=2**17)

    act_func = "tanh"
    kernel = "glorot_uniform"

    x = mom_scaler(p_input)
    x = layers.Flatten()(x)
    m = map_scaler(map_input)
    x = layers.Concatenate()([x, m])
    x = layers.Dense(20, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(40, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(20, activation=act_func, kernel_initializer=kernel)(x)
    outputs = layers.Dense(n_antenna)(x)
    if target_mode == "scaled":
        outputs = coef_scale * tf.math.sinh(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{num_jets}_jets")
    model.add_loss(custom_loss(targets, outputs, antenna_input, born))
    return model

def custom_loss(y_true, y_preds, antennae, born):
    y_pred = tf.math.reduce_sum(tf.math.multiply(y_preds, antennae), axis=1)
    # rescale predictions the same way we did for targets
    if target_mode == "scaled":
        y_pred = y_scaler(tf.math.asinh(tf.math.divide(y_pred, pred_scale))[:, None])
    elif target_mode == "non-scaled":
        y_pred = y_scaler(y_pred[:, None])

    loss = keras.losses.mean_absolute_error(
        tf.math.multiply(y_true, born),
        tf.math.multiply(y_pred, born)
    )
    return loss

def transform_map_variables(map_variables):
    idx = int(map_variables.shape[1] / 2)
    rs = map_variables.numpy()[:, :idx]
    rhos = map_variables.numpy()[:, idx:]
    
    rs = np.log(rs)
    rhos = np.log(rhos-1+1E-10)
    return np.hstack([rs, rhos])
    
def build_base_model(num_jets):
    p_input = keras.Input(shape=(num_jets-1, 4,))
    
    act_func = "tanh"
    kernel = "glorot_uniform"

    mom_scaler = layers.Normalization(axis=1)
    mom_scaler.adapt(X_train[:, 3:], batch_size=2**17)
    x = mom_scaler(p_input)
    x = layers.Flatten()(x)
    x = layers.Dense(20, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(40, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(20, activation=act_func, kernel_initializer=kernel)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=p_input, outputs=outputs, name=f"{num_jets}_jets")
    return model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

w = 1000 # sqrt(s)
num_jets = int(sys.argv[1]) # number of final state jets
train_points = 100000
test_points = 1000000
num_points = train_points + test_points # number of phase-space points to generate
y_global_cut = 0.01 # global phase-space cut
target_dict = {0: "scaled", 1: "non-scaled"}
i = int(sys.argv[2]) # training 20 models, i = {0, ..., 19}
target_mode = target_dict[int(sys.argv[3])] # 0 = rescaling targets, 1 = not rescaling
process_name = "ee_ddx" + "g"*(num_jets-2)

print("Loading data...")
X = np.load(f"/mt/group-batch/gpu-share/htruong/PSData/comparison_sets/PS{num_jets}_0.01_1100k.npz")["PS"]
# res = [born, finite, e1, e2]
res = np.load(f"/mt/group-batch/gpu-share/htruong/PSData/comparison_sets/ME_{process_name}_0.01_1100k.npz")["res"]
born = res[:, 0]
loop = res[:, 1]
kfactors = loop/born
Y = np.array([kfactors, born]).T

X_train, X_test = X[:train_points], X[train_points:]
Y_train, Y_test = Y[:train_points], Y[train_points:]
# shuffle training data but keep testing data fixed as last 90% of data
X_train, Y_train = shuffle(X_train, Y_train, random_state=i)
born_scale = np.percentile(Y_train[:, 1], 0.0)
born_weights = 1 + np.log(Y_train[:, 1] / born_scale)

perms_dict = {
    3: [
        (1,3,2)
        ],
    4: [
        (1,3,2),
        (1,4,2),
        (1,3,4),
        (2,3,4),
    ]
}
perms = perms_dict[num_jets]

_mapper = antennaGenerator.Mapper()
_subantenna = subantenna.SubAntenna(mu=91.188)
_antenna = antenna.Antenna(mu=91.188)
AG = antennaGenerator.AntennaGenerator(perms, _antenna, _mapper)

antennae_born = AG.calculate_inputs(X_train.astype(np.float32), mode="born")
antennae_loop = AG.calculate_inputs(X_train.astype(np.float32), mode="loop")
map_variables = AG.calculate_mapping(X_train.astype(np.float32))
transformed_map_variables = transform_map_variables(map_variables)
antennae_ratio = antennae_loop / antennae_born

# set scales of problem
pred_scale = np.percentile(abs(Y_train[:, 0]), 50.0)
antennae_scale = np.median(antennae_ratio)
coef_scale = pred_scale / antennae_scale
print(f"X shape = {X_train.shape}, Y shape = {Y_train.shape}, antennae shape = {antennae_ratio.shape}")
print(f"pred_scale = {pred_scale}, antennae_scale = {antennae_scale}, coef_scale = {coef_scale}")
print(f"born_scale = {born_scale}")

# add more terms for spin-correlations
if num_jets > 3:
    train_phi_terms = model_inputs.calculate_cs_phis(p=X_train, num_jets=num_jets, cast=True)
    antennae_ratio = tf.concat([antennae_ratio, *train_phi_terms], axis=1)

# model hyperparameters
epochs = 10000
batch_size = 512
lr = 1E-2

y_scaler = layers.Normalization(axis=None)

if target_mode == "scaled":
    scaled = np.arcsinh(Y_train[:, 0] / pred_scale)
    y_scaler.adapt(scaled, batch_size=2**17)
    y_scaled = y_scaler(scaled)
    print(f"Target mode = scaled, sample = {y_scaled[0]}")
    model_suffix = f"{process_name}_NLO_100k_{y_global_cut}_{batch_size}_target_scaled/"
elif target_mode == "non-scaled":
    y_scaler.adapt(Y_train[:, 0], batch_size=2**17)
    y_scaled = y_scaler(Y_train[:, 0])
    print(f"Target mode = non-scaled, sample = {y_scaled[0]}")
    model_suffix = f"{process_name}_NLO_100k_{y_global_cut}_{batch_size}_no_target_scaling/"

checkpoint_path = Path(f"/mt/user-batch/htruong/checkpoints/k_factor_models/04-12/")
checkpoint_path.mkdir(parents=True, exist_ok=True)

if i == 0:
    params = {
            "purpose": f"{num_jets} jets. Test 'new' antennas (get rid of cache). 100k training samples.",
            "num_samples": len(X_train),
            "num_jets": num_jets,
            "d_cut": y_global_cut,
            "num_antennae": antennae_ratio.shape[1],
            "batch_size": batch_size,
            "es_patience": 100,
            "lr_patience": 0,
            "lr": lr,
            "pred_scale": float(pred_scale),
            "antennae_scale": float(antennae_scale),
            "coef_scale": float(coef_scale),
        }
    print(params)
    save_path = checkpoint_path / model_suffix
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "parameters.json", "w") as file:
        json.dump(params, file, indent=4)
else:
    save_path = checkpoint_path / model_suffix
    save_path.mkdir(parents=True, exist_ok=True)

# base_model = build_base_model(num_jets)
antenna_model = build_model(antennae_ratio.shape[1], transformed_map_variables.shape[1], num_jets)

# print("Begin training base model...")
# history_base = train_model(
#     base_model,
#     X_train[:, 3:],
#     y_scaled,
#     checkpoint_path=save_path / f"base_model/model_{i}",
#     epochs=epochs,
#     min_delta=0,
#     loss="mse",
#     batch_size=batch_size,
#     learning_rate=lr,
#     reduce_lr=False,
#     force_save=True
# )

print("Begin training antenna model...")
history_antenna = train_model(
    antenna_model,
    [
        X_train[:, 3:],
        transformed_map_variables,
        antennae_ratio,
        y_scaled,
        born_weights
    ],
    None,
    checkpoint_path=save_path / f"antenna_model/model_{i}",
    epochs=epochs,
    min_delta=0,
    loss=None,
    batch_size=batch_size,
    learning_rate=lr,
    reduce_lr=False,
    force_save=True
)
print("Finished training models...")
