import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
K.set_floatx('float64')
import json
from sklearn.utils import shuffle

from fame.data_generation import cs_dipole, model_inputs
from fame.model.dipole_model_fd import DipoleModel

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

num_jets = 8 # number of final state jets
train_points = 1000000
i = int(sys.argv[1])
mode = int(sys.argv[2])
nodes = int(sys.argv[3])
    

# load data from Timo
print("Loading data...")
data = np.load("/mt/user-batch/htruong/SherpaData/sherpa_z_jets/ps8_me2.npz")
X = data["PS"]
Y = data["ME"]
# filter out negative weights for now
pos_idx = np.where(Y > 0)[0]
# targets = ME*pdf*flux
if mode == 0:
    X = X[pos_idx]
    Y = Y[pos_idx]
# targets = ME
elif mode == 1:
    X = X[pos_idx]
    Y = np.load("/mt/user-batch/htruong/SherpaData/sherpa_z_jets/matrix_elements_recola.npz")["ME"]
    Y = Y[pos_idx]

X_train, X_test = X[:train_points], X[train_points:]
Y_train, Y_test = Y[:train_points], Y[train_points:]
# shuffle training data but keep testing data fixed as last 50% of data
X_train, Y_train = shuffle(X_train, Y_train, random_state=i)

# initialise classes that will generate inputs
relevant_permutations = model_inputs.get_relevant_permutations(num_jets)
CS = cs_dipole.CS_dipole()
relevant_inputs = model_inputs.ModelInputsGenerator(relevant_permutations, CS)

# calculate model inputs
print("Computing model inputs...")
# extra spin-correlation terms
train_phi_terms = model_inputs.calculate_cs_phis(p=X_train, cast=False)
assert train_phi_terms[0].shape == (train_points, 5)
assert train_phi_terms[1].shape == (train_points, 5)

# Catani-Seymour dipoles and recoil factors
# concatenate phi terms with Catani-Seymour dipoles
train_dipoles, train_rfs, train_sijs = relevant_inputs.calculate_inputs(
    p_array=X_train,
    to_concat=[*train_phi_terms]
)
assert train_dipoles.shape == (train_points, 62)
assert train_rfs.shape == (train_points, 52)
assert train_sijs.shape == (train_points, 28)

# set scales of problem
pred_scale = np.float32(np.min(np.abs(Y_train)))
dipole_scale = np.median(train_dipoles)
coef_scale = np.float32(pred_scale / dipole_scale)
print(f"X shape = {X_train.shape}, Y shape = {Y_train.shape}, dipoles shape = {train_dipoles.shape}, ys shape = {train_rfs.shape}")
print(f"pred_scale = {pred_scale}, dipole_scale = {dipole_scale}, coef_scale = {coef_scale}")

# model setup
epochs = 10000
batch_size = 512
lr = 0.0005
# min_delta for EarlyStopping, should go smaller for lower multiplicity as higher accuracy there
min_delta = 1E-6
# turn J off for now, it's not playing nice
J = 0

checkpoint_path = Path(f"/mt/user-batch/htruong/SherpaModels/checkpoints/05-16/Z+4j_tests/")
checkpoint_path.mkdir(parents=True, exist_ok=True)

if i == 0:
    params = {
        "purpose": f"Partonic channel gg>Z>e-e+ggddx, Inputs: PS point, recoil factors, all mandelstams. 800k training samples.",
        "num_samples": len(X_train),
        "num_jets": num_jets,
        "num_dipoles": train_dipoles.shape[1],
        "batch_size": batch_size,
        "es_patience": 60,
        "lr_patience": 30,
        "lr": lr,
        "min_delta": min_delta,
        "J": 0,
        "pred_scale": float(pred_scale),
        "dipole_scale": float(dipole_scale),
        "coef_scale": float(coef_scale)
    }
    save_path = checkpoint_path / f"gg_emepggddx_800k_{nodes}_{batch_size}_pdf/"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "parameters.json", "w") as file:
        json.dump(params, file, indent=4)
else:
    save_path = checkpoint_path / f"gg_emepggddx_800k_{nodes}_{batch_size}_pdf/"
    save_path.mkdir(parents=True, exist_ok=True)

dipole_model = DipoleModel(
    num_jets=num_jets,
    permutations=relevant_permutations,
    X=X_train,
    Y=Y_train,
    recoil_factors=train_rfs,
    mandelstams=train_sijs,
    dipoles=train_dipoles,
    pred_scale=pred_scale,
    coef_scale=coef_scale,
    J=J,
    nodes=nodes
)
dipole_model.preprocess_inputs()
# model for training
dipole_model.build_model()
# model for exporting
dipole_model.build_bare_model()

print("Begin training model...")
history = dipole_model.train_model(
    epochs=epochs,
    checkpoint_path=save_path / f"model_{i}",
    batch_size=batch_size,
    learning_rate=lr,
    min_delta=min_delta
)
print("Finished training model...")

print("Transferring weights to bare model...")
trained_weights = dipole_model.model.get_weights()
dipole_model.bare_model.set_weights(trained_weights[:-3])

print("Testing model against true values...")
coefs, test_rfs, test_sij, test_dipoles, y_preds = dipole_model.dipole_network_predictor(
    model=dipole_model.model,
    inputs=relevant_inputs,
    momenta=X_test,
    model_type="full",
    batch_size=2**16 # try increasing batch size to infer on more points at once
)

coefs2 = dipole_model.bare_model.predict(
    [X_test, test_rfs, test_sij],
    verbose=1,
    batch_size=2**16
)
y_preds2 = tf.reduce_sum(tf.multiply(coefs2, test_dipoles), axis=1).numpy()

print(f"ratio1 = {y_preds/Y_test}, avg_ratio1 = {np.mean(y_preds/Y_test)}")
print(f"ratio1 = {y_preds2/Y_test}, avg_ratio1 = {np.mean(y_preds2/Y_test)}")
assert np.allclose(y_preds, y_preds2)
assert np.allclose(coefs, coefs2)

dipole_model.bare_model.save(save_path / f"bare_model_{i}.h5", include_optimizer=False)
print(f"Finished training model {i}")