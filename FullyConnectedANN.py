# Import Requirements
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def inject_gaussian_noise(magnitude, phase, mean=0.0, std=0.01, magnitude_noise_factor, phase_noise_factor): 
    """
    Inject Gaussian noise into magnitude and phase matrices.
    
    magnitude, phase: np.array of shape (24,24)
    mean: mean of Gaussian noise
    std: standard deviation of noise
    
    Returns:
        noisy_magnitude, noisy_phase: np.array with noise added
    """
    noisy_magnitude = magnitude + magnitude_noise_factor*np.random.normal(loc=mean, scale=std, size=magnitude.shape)
    noisy_phase = phase + phase_noise_factor*np.random.normal(loc=mean, scale=std, size=phase.shape)
    
    return noisy_magnitude, noisy_phase

def load_preprocessed_dataset(scans_folder, plots_folder, num_samples, mag_factor, phase_factor): # Implement this mean the dataset becomes 2x as much
    X = []
    Y = []

    for i in range(1, num_samples + 1):
        # Load original magnitude & phase
        mag_path = os.path.join(scans_folder, f"Scan{i}", f"magnitude{i}_normalized.csv")
        phase_path = os.path.join(scans_folder, f"Scan{i}", f"phase{i}_normalized.csv")
        
        magnitude = np.loadtxt(mag_path, delimiter=",")
        phase = np.loadtxt(phase_path, delimiter=",")
        
        # --- Original scan ---
        scan_input = np.stack([magnitude, phase], axis=-1)
        X.append(scan_input)
        
        # Load corresponding plot
        plot_path = os.path.join(plots_folder, f"Plot{i}", f"plot{i}.png")
        img = load_img(plot_path, target_size=(128, 128), color_mode='rgb')
        img = img_to_array(img) / 255.0
        Y.append(img)
        
        # --- Noisy scan ---
        noisy_mag, noisy_phase = inject_gaussian_noise(magnitude, phase, mean=0.0, std=0.01, \
                                                       magnitude_noise_factor=mag_factor, phase_noise_factor=phase_factor)
        noisy_scan_input = np.stack([noisy_mag, noisy_phase], axis=-1)
        X.append(noisy_scan_input)
        
        # Ground truth remains the same
        Y.append(img)

    # Convert to array
    X = np.array(X)   # shape → (1430, 24, 24, 2)
    Y = np.array(Y)   # shape → (1430, 128, 128, 3)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Train / Val / Test split (80/10/10)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_dataset(scans_folder, plots_folder, num_samples):
    X = []
    Y = []

    # Load data loop
    for i in range(1, num_samples + 1):

        # --- Load magnitude & phase ---
        mag_path = os.path.join(scans_folder, f"Scan{i}", f"magnitude{i}_normalized.csv")
        phase_path = os.path.join(scans_folder, f"Scan{i}", f"phase{i}_normalized.csv") # Consider Normalized Phase

        magnitude = np.loadtxt(mag_path, delimiter=",")
        phase = np.loadtxt(phase_path, delimiter=",")

        # shape → (24, 24, 2)
        scan_input = np.stack([magnitude, phase], axis=-1)
        X.append(scan_input)

        # --- Load ground truth plot image ---
        plot_path = os.path.join(plots_folder, f"Scan{i}.png")

        img = load_img(plot_path, target_size=(128, 128), color_mode='rgb')
        img = img_to_array(img) / 255.0    # normalize

        Y.append(img)

    # Convert to array
    X = np.array(X)   # shape → (715, 24, 24, 2)
    Y = np.array(Y)   # shape → (715, 128, 128, 3)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Train / Val / Test split (80/10/10)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# The fully-connected ANN model
def build_ann():
    model = models.Sequential([
        layers.Input(shape=(24, 24, 2)),
        layers.Flatten(),                         
        layers.Dense(2000, activation='relu'), #Ambrosanio et al. uses 2000 nodes each, can use higher nodes for better accuracy
        layers.Dense(2000, activation='relu'),
        layers.Dense(2000, activation='relu'), #Ambrosanio et al. uses until 8 layers, but took 8 hours to train
        layers.Dense(128 * 128 * 3, activation='linear'),
        layers.Reshape((128, 128, 3))
        # Can reduce number of layers and number of neurons in each dense layer, but will reduce accuracy
        # Can also reduce output size and perform upsampling later
        # Can perform mixed precision training, changing float32 to float16 in GPU
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) #Can decrease learning rate if train and val loss oscillate, and can increase learning rate if loss is coverging too slow
    model.compile(optimizer=optimizer, loss='mse') 
    return model


# Callbacks (EarlyStopping)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,            # stop after 5 epochs with no improvement
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_ann_model.h5',   # save best model
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    verbose=1
)


# Train the model
def train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    model = build_ann()
    model.summary()
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=30,
        batch_size=64,   # Can increase batch size to fasten training, but have to consider changing learning rate
        callbacks=[early_stop, checkpoint, reduceLROnPlateau],
        verbose=1
    )
    test_loss = model.evaluate(X_test, Y_test, verbose=0)
    print("Final Test Loss:", test_loss)
    return model

# Implementation
def pipeline():
    # Path dataset # JANGAN LUPA GANTI
    scans_folder = r"D:\BACKUPDATA\Documents\Projects\IBITec2025\DataSimulation\Scans"
    plots_folder = r"D:\BACKUPDATA\Documents\Projects\IBITec2025\DataSimulation\Plots"
    num_samples = 715
    mag_factor = 0.1 # gaussian noise scaler
    phase_factor = 0.02

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_preprocessed_dataset(scans_folder, plots_folder, num_samples, mag_factor, phase_factor)
    train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    
    #model has been returned from train_model for inference

pipeline()


