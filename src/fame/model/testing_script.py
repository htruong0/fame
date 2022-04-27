import numpy as np
import matplotlib.pyplot as plt

from fame.data_generation import cs_dipole, model_inputs
from fame.utilities import tests
from fame.model.custom_layers import LogLayer, CoefSinhLayer

# Has to be TF 2.8!
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
K.set_floatx('float32')

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

num_jets = 5 # number of final state jets

# load in testing data
testing_data = np.load("testing_data.npz")
# the testing data is the last 20% of the data you provided
X_test = testing_data["PS"]
Y_test = testing_data["ME"]
# load these to check against
loaded_C = testing_data["C_preds"]
loaded_dipoles = testing_data["test_dipoles"]
loaded_recoils = testing_data["test_recoils"]

# compute model inputs
relevant_permutations = model_inputs.get_relevant_permutations(num_jets)
tests.check_relevant_permutations(relevant_permutations, num_jets)

# initialise classes that will generate inputs
CS = cs_dipole.CS_dipole(mode='gluon')
relevant_inputs = model_inputs.ModelInputsGenerator(relevant_permutations, CS, cast=True)

test_phi_terms = model_inputs.calculate_cs_phis(p=X_test, num_jets=num_jets, cast=True)
tests.check_phi_terms(test_phi_terms, num_jets)

# Catani-Seymour dipoles and recoil factors
# concatenate phi terms with Catani-Seymour dipoles
test_dipoles, test_ys = relevant_inputs.calculate_inputs(
    p_array=X_test,
    to_concat=[*test_phi_terms]
)

tests.check_recoil_factors(test_ys, num_jets)
tests.check_all_dipoles(test_dipoles, num_jets)

# check saved ones are the same as re-computed ones
assert np.allclose(test_dipoles, loaded_dipoles)
assert np.allclose(test_ys, loaded_recoils)

# load in individual model (picked first but shouldn't make a difference) and ensemble model
individual_model = keras.models.load_model("h5_models/ee_uuxggg_model.0.h5")
ensemble_model = keras.models.load_model("h5_models/ee_uuxggg_ensemble_model.h5")

coef_preds_individual = individual_model.predict([X_test[:, 2:], test_ys], batch_size=2**16, verbose=1)
coef_preds_ensemble = ensemble_model.predict([X_test[:, 2:], test_ys], batch_size=2**16, verbose=1)

# I saved the predictions from the ensemble model so can compare here to check everything is correct
assert np.allclose(coef_preds_ensemble, loaded_C)

# combine coefficients with dipoles to get matrix element
y_preds_individual = np.sum(coef_preds_individual*test_dipoles, axis=1)
y_preds_ensemble = np.sum(coef_preds_ensemble*test_dipoles, axis=1)


diff = lambda y_true, y_pred: abs(y_true - y_pred) / y_true * 100
ratio = lambda y_true, y_pred: y_pred / y_true

# make quick plots to compare with true values
bins = np.linspace(0.985, 1.015, 50)
logbins = np.logspace(-4, 1, 50)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(diff(Y_test, y_preds_individual), bins=logbins, alpha=0.5, label="individual")
ax[0].hist(diff(Y_test, y_preds_ensemble), bins=logbins, alpha=0.5, label="20 model ensemble")
ax[1].hist(ratio(Y_test, y_preds_individual), bins=bins, alpha=0.5, label="individual")
ax[1].hist(ratio(Y_test, y_preds_ensemble), bins=bins, alpha=0.5, label="20 model ensemble")
ax[0].set_xscale("log")
ax[0].set_xlabel("Abs % difference")
ax[1].set_xlabel("pred/truth ratio")
for axis in ax:
    axis.legend()
    axis.set_ylabel("Frequency")
plt.tight_layout()
plt.show()
