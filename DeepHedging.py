# =============================================================================
# Reproducibility setup
# -----------------------------------------------------------------------------
# The following configuration enforces deterministic behavior across Python,
# NumPy and TensorFlow to ensure full reproducibility of numerical results.
#
#
# IMPORTANT:
# PYTHONHASHSEED=1 must be set BEFORE starting Python, otherwise hash-based
# operations may still introduce randomness.
#
#
# NOTE:
# - This significantly reduces performance.
# - Use primarily for experiments, comparisons and thesis results.
# - For production or large-scale training, consider relaxing these settings.
# =============================================================================

import os
seed = 1

# TensorFlow determinism
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# CPU thread pools (NumPy/SciPy/BLAS)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# CPU kernel selection stability 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import tensorflow as tf


tf.config.experimental.enable_op_determinism()

random.seed(seed)

np.random.seed(seed)
tf.random.set_seed(seed)


from tensorflow.keras.layers import Input, Dense, TimeDistributed, Subtract, Lambda, Flatten, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.io import loadmat, savemat



############################
# Setup
############################

# case and measure specification
Case = input("Case (Case_1 / Case_2): ").strip()
Q= input("Measure (Q / Q_star): ").strip()

# trading time grids
N_sim=[50,100,250]

# setup
T=1
S_0=100


###############################################################################
# NN Training and Testing
###############################################################################

for N in N_sim:
    
    
    ###########################################################################
    # NN Training 
    ###########################################################################
    
   # load MATLAB training data 
    filename = "data_NN_training_" + Case + "_" + Q + "_N_" + str(N) + ".mat"
    training_data = loadmat(filename)
    training_paths = training_data['paths_NN_training']
    training_V_t   = training_data['V_t_NN_training']
    
    # reshape training paths and V_t in NN compatible format
    training_paths = np.transpose(training_paths, (2, 0, 1))   # (M, N+1, 3) columns: (t,S,v)
    training_V_t   = np.transpose(training_V_t, (2, 0, 1))     # (M, N+1, 1) or (M, N+1)
    
    M   = training_paths.shape[0]
    Np1 = training_paths.shape[1]
    N   = Np1 - 1
    
    # hyperparameters specification
    activator = "tanh"
    d = 2          # hidden layers
    n = 200        # width
    epochs=60
    batch_size = 128
    
    # optimizer specification
    learning_rate=3e-4
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,clipnorm=1.0)

    # prepare training inputs (Obs at t decision times t0..t_{N-1})
    Obs_np = training_paths[:, :N, :].astype(np.float32)     # (M, N, 3)
    Obs_np[:,:,0]=T-Obs_np[:,:,0]
    Obs_np[:,:,1]=np.log(Obs_np[:,:,1]/S_0)
    
    
    # dS increments (already in discounted units)
    S = training_paths[:, :, 1]                         # (M, N+1)
    dS = (S[:, 1:] - S[:, :-1]).astype(np.float32)      # (M, N)
    Incr_np = dS[:, :, None]                            # (M, N, 1)
    
    # discounted exogenous value process (from ADI pricing)
    Vex_np = training_V_t.astype(np.float32)            # (M, N+1, 1)
    
    
    ############################
    # Define hedge network (phi)
    ############################
    
    # architecture of the hedge network --- expecting (t,S,v) vector
    g_input = Input(shape=(3,))    # (t,S,v)
    
    output = g_input
    for i in range(d):
        output = Dense(n, activation=activator)(output)
    phi= Dense(1, activation='linear')(output)
        
    hedge = Model(inputs=g_input, outputs=phi)
    print("\n[Below] Network for the hedging position (phi):")
    hedge.summary()
    
    
    ############################
    # Define training model  --  wrt to C_T only (CT_fast)
    ############################
    Obs_in  = Input(shape=(N, 3))
    Incr_in = Input(shape=(N, 1))
    Vex_in  = Input(shape=(N+1, 1))
    
    inputs = [Obs_in, Incr_in, Vex_in]
    
    # phi_k = g_theta(state_k)
    Phi = TimeDistributed(hedge)(Obs_in)                # (batch, N, 1)
    
    # flatten Phi and Incr for vectorized Gain computation
    Phi_flat=Flatten()(Phi)                             #(batch, N)
    Incr_flat=Flatten()(Incr_in)                        #(batch,N)
    
    # terminal trading gain
    Gain=Dot(axes=1)([Phi_flat,Incr_flat])              #(batch,1)
    
    # terminal exogenous value increment: V_T - V_0 (in discounted units)
    V0 = Lambda(lambda x: x[:, 0, :])(Vex_in)           # (batch, 1)
    VT = Lambda(lambda x: x[:, -1, :])(Vex_in)          # (batch, 1)
    dV_T = Subtract()([VT, V0])                         # (batch, 1)
    
    # terminal cost: C_T = (V_T - V_0) - Gain
    CT_fast = Subtract()([dV_T, Gain])                  # (batch, 1)
    
    # compile model
    model_CT = Model(inputs=inputs, outputs=CT_fast)
    model_CT.compile(optimizer=optimizer, loss='mean_squared_error')
    print("\n[Below] FAST model for terminal cost C_T (objective is E[C_T^2]):")
    model_CT.summary()
    
    
    # define callbacks --  in particular restore_best_weights
    callbacks = [
       ReduceLROnPlateau(
           monitor="loss",
           factor=0.5,
           patience=8,
           min_lr=1e-6,
           verbose=1
       ),
       EarlyStopping(
           monitor="loss",
           patience=15,
           min_delta=1e-4,
           restore_best_weights=True,
           verbose=1
       ),
    ]
    
    # ckpt_path = f"tmp_{Case}_{Q}_N_{N}.weights.h5"
    
    # callbacks = [
    # ModelCheckpoint(
    #     filepath=ckpt_path,
    #     monitor="loss",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     mode="min",
    #     verbose=1)]

    

    
    ############################
    # Define model all (full path C_t, dC, Phi, C_T) - run after training for full diagnostics
    ############################
    
    # per-step gain increments
    Gain_incr = Lambda(lambda z: z[0] * z[1])([Phi, Incr_in])   # (batch, N, 1)
    
    # per-step exogenous value increments
    dV = Lambda(lambda x: x[:, 1:, :] - x[:, :-1, :])(Vex_in)   # (batch, N, 1)
    
    # per-step cost increments: dC_{k+1} = dV_{k+1} - phi_k*dS_{k+1}
    dC = Subtract()([dV, Gain_incr])                            # (batch, N, 1)
    
    # cumulative cost path: C_0=0, C_{k+1} = sum_{i=0}^k dC_{i+1}
    C_path = Lambda(lambda dc: tf.concat([tf.zeros_like(dc[:, :1, :]), tf.cumsum(dc, axis=1)], axis=1))(dC)                                                                     # (batch, N+1, 1)
    
    CT_path = Lambda(lambda c: c[:, -1, :])(C_path)              # (batch, 1)
    
    model_all = Model(inputs=inputs, outputs=[C_path, dC, Phi, CT_path])
    
    
    ############################
    # Train (target is zero => loss = mean(CT^2))
    ############################
    
    # preallocate output
    y0 = np.zeros((M, 1), dtype=np.float32)
    
    # fit model
    model_CT.fit([Obs_np, Incr_np, Vex_np], y0,epochs=epochs, batch_size=batch_size,callbacks=callbacks,shuffle=True,verbose=1)
    
   # model_CT.load_weights(ckpt_path)   # <-- BESTE weights wiederherstellen
    
    # in-sample evaluation for sanity checks (not exported)
    C_path_pred, dC_pred, Phi_pred, CT_pred = model_all.predict([Obs_np, Incr_np, Vex_np])
    
    print("C_path_pred_training:", C_path_pred.shape)            # (M, N+1, 1)
    print("dC_pred_training:", dC_pred.shape)                    # (M, N, 1)
    print("Phi_pred:_training", Phi_pred.shape)                  # (M, N, 1)
    print("CT_pred_training:", CT_pred.shape)                    # (M, 1)
    print("Mean CT training:", CT_pred.mean(), "Std CT training:", CT_pred.std())
    
    
    ###########################################################################
    # NN Testing
    ###########################################################################
    
    ############################
    # Prediction for MATLAB export
    ############################
    
    # load MATLAB training data 
    filename = "data_NN_testing_" + Case + "_" + Q + "_N_" + str(N) + ".mat"
    testing_data = loadmat(filename)
    testing_paths = testing_data['paths_NN_testing']
    testing_V_t   = testing_data['V_t_NN_testing']
    
    # reshape testing paths and V_t in NN compatible format
    testing_paths = np.transpose(testing_paths, (2, 0, 1))   # (M, N+1, 3) columns: (t,S,v)
    testing_V_t   = np.transpose(testing_V_t, (2, 0, 1))     # (M, N+1, 1) or (M, N+1)
    
    # Prepare testing inputs (Obs at t decision times t0..t_{N-1})
    Obs_np = testing_paths[:, :N, :].astype(np.float32) # (M, N, 3)
    Obs_np[:,:,0]=T-Obs_np[:,:,0]
    Obs_np[:,:,1]=np.log(Obs_np[:,:,1]/S_0)
    
    # dS increments (already in discounted units)
    S = testing_paths[:, :, 1]                          # (M, N+1)
    dS = (S[:, 1:] - S[:, :-1]).astype(np.float32)      # (M, N)
    Incr_np = dS[:, :, None]                            # (M, N, 1)
    
    # discounted exogenous value process (from ADI pricing)
    Vex_np = testing_V_t.astype(np.float32)
    
    # out-of-sample evaluation using test data
    C_path_pred, dC_pred, Phi_pred, CT_pred = model_all.predict([Obs_np, Incr_np, Vex_np])
    
    print("C_path_pred:", C_path_pred.shape)            # (M, N+1, 1)
    print("dC_pred:", dC_pred.shape)                    # (M, N, 1)
    print("Phi_pred:", Phi_pred.shape)                  # (M, N, 1)
    print("CT_pred:", CT_pred.shape)                    # (M, 1)
    print("Mean CT:", CT_pred.mean(), "Std CT:", CT_pred.std())
    
    # realized discounted cost increments for MATLAB export
    dC_mat=np.transpose(dC_pred,(1,2,0))
    
    filename = "dC_pred_" + Case + "_" + Q + "_N_" + str(N) + ".mat"
    savemat(filename,{"dC_NN":dC_mat})








