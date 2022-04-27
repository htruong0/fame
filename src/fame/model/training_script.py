import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import json
from sklearn.utils import shuffle

from fame.utilities import tests
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

w = 1000 # sqrt(s)
num_jets = 5 # number of final state jets
train_points = 800000
test_points = 200000
num_points = train_points + test_points # number of phase-space points to generate
y_global_cut = 0.01 # global phase-space cut
i = int(sys.argv[1]) # training 20 models, i = {0, ..., 19}

# load data from Timo
print("Loading data...")
data = np.load("/mt/group-batch/gpu-share/htruong/Data/5jet/sherpa_data_emep_uuxggg.npz")
X = data["PS"]
Y = data["ME"]

X_train, X_test = X[:train_points], X[train_points:]
Y_train, Y_test = Y[:train_points], Y[train_points:]
# shuffle training data but keep testing data fixed as last 20% of data
X_train, Y_train = shuffle(X_train, Y_train, random_state=i)


relevant_permutations = model_inputs.get_relevant_permutations(num_jets)
tests.check_relevant_permutations(relevant_permutations, num_jets)

# initialise classes that will generate inputs
CS = cs_dipole.CS_dipole(mode='gluon')
relevant_inputs = model_inputs.ModelInputsGenerator(relevant_permutations, CS, cast=False)

# calculate model inputs
print("Computing model inputs...")
# extra spin-correlation terms
train_phi_terms = model_inputs.calculate_cs_phis(p=X_train, num_jets=num_jets, cast=False)
tests.check_phi_terms(train_phi_terms, num_jets)

# Catani-Seymour dipoles and recoil factors
# concatenate phi terms with Catani-Seymour dipoles
train_dipoles, train_ys = relevant_inputs.calculate_inputs(
    p_array=X_train,
    to_concat=[*train_phi_terms]
)

tests.check_recoil_factors(train_ys, num_jets)
tests.check_all_dipoles(train_dipoles, num_jets)

# set scales of problem
pred_scale = np.min(Y_train)
dipole_scale = np.mean(train_dipoles)
coef_scale = pred_scale / dipole_scale
print(f"X shape = {X_train.shape}, Y shape = {Y_train.shape}, dipoles shape = {train_dipoles.shape}, ys shape = {train_ys.shape}")
print(f"pred_scale = {pred_scale}, dipole_scale = {dipole_scale}, coef_scale = {coef_scale}")

# model setup
epochs = 10000
batch_size = 512
lr = 0.001
# min_delta for EarlyStopping, should go smaller for lower multiplicity as higher accuracy there
min_delta = 1E-8
# J is tuned manually such that f_pen << mse
J = 1E14

checkpoint_path = Path(f"/mt/user-batch/htruong/SherpaModels/checkpoints/02-23/{num_jets}jet_{y_global_cut}/")
checkpoint_path.mkdir(parents=True, exist_ok=True)

if i == 0:
    params = {
        "purpose": f"{num_jets} jets, CS dipoles. Logged targets. 800k training samples.",
        "num_samples": len(X_train),
        "num_jets": num_jets,
        "d_cut": y_global_cut,
        "num_dipoles": train_dipoles.shape[1],
        "batch_size": batch_size,
        "es_patience": 40,
        "lr_patience": 20,
        "lr": lr,
        "J": J,
        "pred_scale": float(pred_scale),
        "dipole_scale": float(dipole_scale),
        "coef_scale": float(coef_scale)
    }
    save_path = checkpoint_path / f"ee_uuxggg_800k_{y_global_cut}_{batch_size}/"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "parameters.json", "w") as file:
        json.dump(params, file, indent=4)
else:
    save_path = checkpoint_path / f"ee_uuxggg_800k_{y_global_cut}_{batch_size}/"
    save_path.mkdir(parents=True, exist_ok=True)

dipole_model = DipoleModel(
    num_jets=num_jets,
    permutations=relevant_permutations,
    X=X_train,
    Y=Y_train,
    recoil_factors=train_ys,
    dipoles=train_dipoles,
    pred_scale=pred_scale,
    coef_scale=coef_scale,
    J=J
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
coefs, test_ys, test_dipoles, y_preds = dipole_model.dipole_network_predictor(
    model=dipole_model.model,
    inputs=relevant_inputs,
    momenta=X_test,
    batch_size=2**16 # try increasing batch size to infer on more points at once
)

coefs2 = dipole_model.bare_model.predict(
    [X_test[:, 2:], test_ys],
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