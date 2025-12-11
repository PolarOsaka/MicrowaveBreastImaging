# Import Requirements
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import psutil
import random
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from keras_tuner import Hyperband
from sklearn.model_selection import train_test_split

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

        img = load_img(plot_path, target_size=(128, 128), color_mode='grayscale')
        img = img_to_array(img) / 255.0    # normalize

        Y.append(img)

    # Convert to array
    X = np.array(X)   # shape → (715, 24, 24, 2)
    Y = np.array(Y)   # shape → (715, 128, 128, 1)

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
def build_ann(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(24,24,2)))

    # Tuning jumlah layer
    for i in range(hp.Int("num_layers", 2, 6)):
        model.add(Dense(
            units=hp.Int("units_" + str(i), 500, 3000, step=500),
            activation="relu"
        ))

    model.add(Dense(128*128*1, activation="sigmoid"))
    model.add(Reshape((128,128,1)))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("lr", 1e-5, 1e-3, sampling="log")
        ),
        loss="mse"
    )
    return model


tuner = Hyperband(
    build_ann,
    objective='val_loss',
    max_epochs=30,            
    factor=3,
    directory='tuning_results',
    project_name='microwave_ann'
)

# Callbacks
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
    

# Define class for monitoring object
class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.memory_usage = []
        self._proc = psutil.Process(os.getpid())

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Loss
        self.train_losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss", None))

        # Timing
        self.epoch_times.append(time.time() - getattr(self, "_epoch_start_time", time.time()))

        # Memory
        mem = self._proc.memory_info().rss / (1024 ** 2)
        self.memory_usage.append(mem)

    def on_train_end(self, logs=None):
        print("Training finished. Epochs recorded:", len(self.train_losses))


# Train the model
def train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, monitor, best_model):
    
    # Build ANN
    model = best_model  # Sequential model
    model.summary()

    # Train
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop, checkpoint, reduceLROnPlateau, monitor],
        verbose=1
    )

    # Test
    test_loss = model.evaluate(X_test, Y_test, verbose=0)
    print("Final Test Loss:", test_loss)
    return model, monitor


def visualize_result(monitor):
    plt.figure(figsize=(12, 10))

    # Subplot 1
    plt.subplot(1, 3, 1)
    plt.plot(monitor.train_losses, label="Train Loss")
    plt.plot(monitor.val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # Subplot 2
    plt.subplot(1, 3, 2)
    plt.plot(monitor.epoch_times)
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Training Time per Epoch")
    plt.grid(True)

    # Subplot 3
    plt.subplot(1, 3, 3)
    plt.plot(monitor.memory_usage)
    plt.xlabel("Epoch")
    plt.ylabel("Memory Usage (MB)")
    plt.title("CPU Memory Usage per Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def show_random_comparisons(model, X_val, Y_val, num):
    
    # pilih index random
    indices = random.sample(range(len(X_val)), num)

    # Buat figure besar
    plt.figure(figsize=(8, num * 1))  # tinggi mengikuti jumlah sampel

    for i, idx in enumerate(indices):
        x = X_val[idx:idx+1]          # (1, 24, 24, 2)
        y_true = Y_val[idx]           # (128, 128, 1)

        # Prediksi
        y_pred = model.predict(x)[0]   # (128, 128, 1)

        # Difference map
        diff = np.abs(y_true - y_pred)

        # --- Plot 3 kolom: GT | Pred | Diff ---
        # Row i, col 1 → GT
        plt.subplot(num, 3, i*3 + 1)
        plt.imshow(y_true)
        #plt.title("Ground Truth")
        plt.axis("off")

        # Row i, col 2 → Prediction
        plt.subplot(num, 3, i*3 + 2)
        plt.imshow(np.clip(y_pred, 0, 1), cmap='turbo')
        #plt.title("Reconstructed")
        plt.axis("off")

        # Row i, col 3 → Difference
        plt.subplot(num, 3, i*3 + 3)
        plt.imshow(diff, cmap="inferno")
        #plt.title("Difference")
        plt.axis("off")

    plt.suptitle("Ground Truth vs Reconstruction vs Differences")
    plt.tight_layout()
    plt.show()


# Implementation
def pipeline():
    # Path dataset # JANGAN LUPA GANTI
    scans_folder = r"D:\BACKUPDATA\Documents\Projects\IBITec2025\DataSimulation\Scans"
    plots_folder = r"D:\BACKUPDATA\Documents\Projects\IBITec2025\DataSimulation\Plots"
    num_samples = 715
    mag_factor = 0.1 # gaussian noise scaler
    phase_factor = 0.02
    monitor = TrainingMonitor()
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_dataset(scans_folder, plots_folder, num_samples)
    #X_train, Y_train, X_val, Y_val, X_test, Y_test = load_preprocessed_dataset(scans_folder, plots_folder, num_samples, mag_factor, phase_factor)
    
    tuner.search(X_train, Y_train, epochs=20, validation_data=(X_val, Y_val), callbacks=[early_stop, reduceLROnPlateau, checkpoint])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best number of layers:", best_hps.get('num_layers'))
    for i in range(best_hps.get('num_layers')):
        print(f"  Layer {i}: units = {best_hps.get('units_' + str(i))}")
    print("Best learning rate:", best_hps.get('lr'))
    
    best_model = tuner.hypermodel.build(best_hps)

    model, monitor_result = train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, monitor, best_model)
    visualize_result(monitor_result)
    #model has been returned from train_model for inference
    show_random_comparisons(model, X_val, Y_val, num=15)


pipeline()


