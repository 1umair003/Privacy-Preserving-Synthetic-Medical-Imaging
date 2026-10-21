"""Hyperparameters taken from the manuscript (Sections 3.1-3.9)."""

SEED = 42

# ---------------------------------------------------------------- classes
# Folder names expected under every data directory (one sub-folder per class).
CLASSES = [
    "Bacterial_Pneumonia",
    "COVID-19",
    "Normal",
    "Tuberculosis",
    "Viral_Pneumonia",
]
NUM_CLASSES = len(CLASSES)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---------------------------------------------------------------- CNN training (Sec. 3.9)
EPOCHS = 30                 # T in the cosine schedule (Eq. 28)
BATCH_SIZE = 32
INITIAL_LR = 1e-4           # eta_0
N_FOLDS = 5                 # stratified k-fold
FINE_TUNE_LAST_N_LAYERS = 100   # pre-trained setting: last 100 layers trainable

# ReduceLROnPlateau behaviour layered on top of the cosine schedule (Fig. 2)
PLATEAU_FACTOR = 0.2
PLATEAU_PATIENCE = 3

# ---------------------------------------------------------------- classification head (Sec. 3.6)
DENSE_UNITS = (512, 256)
DROPOUT_RATES = (0.5, 0.3)

# ---------------------------------------------------------------- input size per backbone (Sec. 3.1)
INPUT_SIZE = {
    "ResNet50": 224,
    "InceptionV3": 299,
    "Xception": 299,
}

# ---------------------------------------------------------------- online augmentation (Sec. 3.2)
AUGMENTATION = {
    "rotation_degrees": 30,
    "shift": 0.3,
    "shear": 0.3,
    "zoom": 0.3,
    "brightness": (0.8, 1.2),
    "horizontal_flip": True,
}

# ---------------------------------------------------------------- LoRA-SDXL (Sec. 3.4)
LORA = {
    "rank": 128,
    "alpha": 128,
    "epochs": 50,
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "batch_size": 16,
    "resolution": 1024,
    "ddim_steps": 200,
    "ssim_filter_threshold": 0.85,
    "oversample_classes": ("Tuberculosis", "COVID-19"),
    "oversample_factor": 2,
}

# Text prompts used to condition SDXL for each class.
CLASS_PROMPTS = {
    "Bacterial_Pneumonia": (
        "a frontal chest X-ray radiograph showing bacterial pneumonia, "
        "dense lobar consolidation, air bronchograms, high resolution, grayscale"
    ),
    "COVID-19": (
        "a frontal chest X-ray radiograph showing COVID-19 pneumonia, bilateral "
        "peripheral lower-zone ground-glass opacities, high resolution, grayscale"
    ),
    "Normal": (
        "a frontal chest X-ray radiograph of healthy normal lungs, clear lung "
        "fields, sharp costophrenic angles, high resolution, grayscale"
    ),
    "Tuberculosis": (
        "a frontal chest X-ray radiograph showing pulmonary tuberculosis, "
        "upper-lobe nodular infiltrates with cavitation, high resolution, grayscale"
    ),
    "Viral_Pneumonia": (
        "a frontal chest X-ray radiograph showing viral pneumonia, bilateral "
        "interstitial reticular infiltrates, perihilar haziness, high resolution, grayscale"
    ),
}
