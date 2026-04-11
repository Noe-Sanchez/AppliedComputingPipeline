import cv2
import numpy as np
import torch
import pandas as pd
from pycparser.c_ast import Return
from urllib3 import Retry

from depth_anything_v2.dpt import DepthAnythingV2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from tqdm import tqdm
import time
import tifffile as tiff

# WARP IMAGES
# Pass the path for the RGB image, TIF image, directory where the images are, directory for where to save
def warp_image(img_rgb_path, img_tif_path, images_folder, save_folder):
    img_rgb = cv2.imread(f"{images_folder}/{img_rgb_path}")
    img_tif = cv2.imread(f"{images_folder}/{img_tif_path}")

    gray1 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_tif, cv2.COLOR_BGR2GRAY)

    # Detect features
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use best matches
    good_matches = matches[:500]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography: warp img_rgb to img_tif
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    h, w = img_tif.shape[:2]
    warped_rgb = cv2.warpPerspective(img_rgb, H, (w, h))

    # Blend to inspect overlap
    overlay = cv2.addWeighted(warped_rgb, 0.5, img_tif, 0.5, 0)

    #cv2.imshow("Warped img1", warped_rgb)
    #cv2.imshow("Overlay", overlay)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(f"{save_folder}/img_rgb_path_{img_rgb_path}", warped_rgb)
    cv2.imwrite(f"{save_folder}/img_tif_path_{img_tif_path}", img_tif)


# Test warp_image
#img1 = "DJI_20260216171445_0021_D.JPG"
#img2 = "DJI_20260216171445_0021_MS_R.TIF"

#warp_image(img1, img2, "../data/training/original_images", "../data/trining/warped_images/test_save_img")



# IMAGES TO CVS
def make_patch_column_names(radius: int):
    """
    Create column names for a square patch centered at (0,0).
    Example for radius=1:
        R_m1_m1, G_m1_m1, B_m1_m1, ..., R_0_0, G_0_0, B_0_0, ..., NIR
    """
    col_names = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dy_name = f"m{abs(dy)}" if dy < 0 else f"p{dy}"
            dx_name = f"m{abs(dx)}" if dx < 0 else f"p{dx}"

            # optional prettier center name
            if dy == 0 and dx == 0:
                suffix = "centro"
            else:
                suffix = f"{dy_name}_{dx_name}"

            col_names.extend([
                f"R_{suffix}",
                f"G_{suffix}",
                f"B_{suffix}"
            ])

    col_names.append("NIR")
    return col_names


def extract_patch_features(rgb_img, nir_img, valid_mask, radius):
    """
    Extract all patch features for every valid pixel at once using NumPy.
    Returns:
        data: (N, num_features+1)
        rows, cols: coordinates of valid center pixels
    """
    h, w, _ = rgb_img.shape

    # Exclude borders according to patch radius
    valid = valid_mask.copy()
    valid[:radius, :] = False
    valid[-radius:, :] = False
    valid[:, :radius] = False
    valid[:, -radius:] = False

    rows, cols = np.where(valid)

    feature_blocks = []

    # Loop over patch offsets, but not over every pixel
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            patch_pixels = rgb_img[rows + dy, cols + dx]   # shape (N, 3)
            feature_blocks.append(patch_pixels)

    # Concatenate all RGB features
    X = np.concatenate(feature_blocks, axis=1)  # shape (N, patch_pixels*3)

    # NIR target
    y = nir_img[rows, cols].reshape(-1, 1)

    data = np.concatenate([X, y], axis=1)
    return data, rows, cols


def generate_cvs_image(RGB_PATH, NIR_PATH, OUTPUT_CSV, PATCH_RADIUS=2):
    rgb_frame = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)
    nir_frame = cv2.imread(NIR_PATH, cv2.IMREAD_UNCHANGED)

    if rgb_frame is None:
        raise FileNotFoundError(f"Could not read RGB image: {RGB_PATH}")

    if nir_frame is None:
        raise FileNotFoundError(f"Could not read NIR image: {NIR_PATH}")

    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    if nir_frame.ndim == 3:
        nir_frame = nir_frame[:, :, 0]

    print("RGB shape:", rgb_frame.shape)
    print("NIR shape:", nir_frame.shape)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'../data/models/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    frame_bgr = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)

    blur = cv2.blur(frame_bgr, (20, 20))
    imgHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 18, 0], dtype=np.uint8)
    upper = np.array([76, 255, 255], dtype=np.uint8)

    color_mask = cv2.inRange(imgHsv, lower, upper)

    depth = model.infer_image(frame_bgr)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    _, depth_mask = cv2.threshold(depth_norm, 100, 255, cv2.THRESH_BINARY)

    both = cv2.bitwise_and(depth_mask, color_mask)

    patch_size = 2 * PATCH_RADIUS + 1

    valid_mask = (both == 255)

    data, rows, cols = extract_patch_features(
        rgb_img=rgb_frame,
        nir_img=nir_frame,
        valid_mask=valid_mask,
        radius=PATCH_RADIUS
    )

    col_names = make_patch_column_names(PATCH_RADIUS)
    full = pd.DataFrame(data, columns=col_names)

    full.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved as: {OUTPUT_CSV}")



#RGB_PATH = '../data/training/warped_images/warped_for_red_1.jpg'
#NIR_PATH = '../data/training/warped_images/red_1.TIF'
#OUTPUT_CSV = '../data/training/generated_CSVs/training_red_1.csv'
#generate_cvs_image(RGB_PATH, NIR_PATH, OUTPUT_CSV)




# TRAIN MODEL
def train_model(path_to_dataset, path_to_model, CHUNK_SIZE=1_000_000, TOTAL_ROWS = 24_335_559):
    print(f"Loading {path_to_dataset}...")
    chunks = []

    with tqdm(total=TOTAL_ROWS, unit="filas", desc="Loading CSV") as pbar:
        for chunk in pd.read_csv(path_to_dataset, dtype='float32', chunksize=CHUNK_SIZE):
            chunks.append(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunks, axis=0)
    print(f"Loading complete. Initial rows: {len(df)}")

    print("Cleaning Data...")
    df = df[(df['NIR'] < 255) & (df['NIR'] > 0)].copy()

    df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)

    rgb_feature_cols = [c for c in df.columns if
                        c != "NIR" and (c.startswith("R_") or c.startswith("G_") or c.startswith("B_"))]

    R_cols = sorted([c for c in rgb_feature_cols if c.startswith("R_")])
    G_cols = sorted([c for c in rgb_feature_cols if c.startswith("G_")])
    B_cols = sorted([c for c in rgb_feature_cols if c.startswith("B_")])

    if not (len(R_cols) == len(G_cols) == len(B_cols)):
        raise ValueError("The number of R, G y B columns isn't consistent.")

    positions = [c[2:] for c in R_cols]

    if "centro" not in positions:
        raise ValueError("No 'center' position found in the CSV.")

    print("Creating new characteristics...")
    eps = 1e-6

    R = df['R_centro']
    G = df['G_centro']
    B = df['B_centro']

    sum_rgb = R + G + B + eps

    df['GLI_centro'] = (2 * G - R - B) / (2 * G + R + B + eps)
    df['ExG_centro'] = 2 * G - R - B
    df['VARI_centro'] = (G - R) / (G + R - B + eps)

    df['r_norm_centro'] = R / sum_rgb
    df['g_norm_centro'] = G / sum_rgb
    df['b_norm_centro'] = B / sum_rgb

    for pos in positions:
        Rn = df[f'R_{pos}']
        Gn = df[f'G_{pos}']
        Bn = df[f'B_{pos}']

        df[f'GLI_{pos}'] = (2 * Gn - Rn - Bn) / (2 * Gn + Rn + Bn + eps)
        df[f'ExG_{pos}'] = 2 * Gn - Rn - Bn

    df['R_mean_patch'] = df[R_cols].mean(axis=1)
    df['G_mean_patch'] = df[G_cols].mean(axis=1)
    df['B_mean_patch'] = df[B_cols].mean(axis=1)

    df['R_std_patch'] = df[R_cols].std(axis=1)
    df['G_std_patch'] = df[G_cols].std(axis=1)
    df['B_std_patch'] = df[B_cols].std(axis=1)

    df['R_min_patch'] = df[R_cols].min(axis=1)
    df['G_min_patch'] = df[G_cols].min(axis=1)
    df['B_min_patch'] = df[B_cols].min(axis=1)

    df['R_max_patch'] = df[R_cols].max(axis=1)
    df['G_max_patch'] = df[G_cols].max(axis=1)
    df['B_max_patch'] = df[B_cols].max(axis=1)

    GLI_cols = [f'GLI_{pos}' for pos in positions]
    ExG_cols = [f'ExG_{pos}' for pos in positions]

    df['GLI_mean_patch'] = df[GLI_cols].mean(axis=1)
    df['GLI_std_patch'] = df[GLI_cols].std(axis=1)

    df['ExG_mean_patch'] = df[ExG_cols].mean(axis=1)
    df['ExG_std_patch'] = df[ExG_cols].std(axis=1)

    df['R_center_minus_mean'] = df['R_centro'] - df['R_mean_patch']
    df['G_center_minus_mean'] = df['G_centro'] - df['G_mean_patch']
    df['B_center_minus_mean'] = df['B_centro'] - df['B_mean_patch']

    df['GLI_center_minus_mean'] = df['GLI_centro'] - df['GLI_mean_patch']
    df['ExG_center_minus_mean'] = df['ExG_centro'] - df['ExG_mean_patch']

    df['G_R_ratio_centro'] = df['G_centro'] / (df['R_centro'] + eps)
    df['G_B_ratio_centro'] = df['G_centro'] / (df['B_centro'] + eps)
    df['R_B_ratio_centro'] = df['R_centro'] / (df['B_centro'] + eps)

    print("New characteristics created")
    print(f"Total Column Number: {len(df.columns)}")

    X = df.drop(["NIR"], axis=1)
    y = df["NIR"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    print("Creating RandomForestRegressor...")
    NIR_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=8,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    print("Training fit model...")
    start_time = time.time()
    NIR_model.fit(X_train, y_train)
    end_time = time.time()

    print(f"Training complete in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Saving model as {path_to_model}...")
    joblib.dump(NIR_model, path_to_model)



#path_to_csv_file = "red_dataset.csv"   # <-- your new 5x5 CSV
#path_to_model_file = "R_random_forest_5x5_features_50p_v1.joblib"
#train_model(path_to_csv_file, path_to_model_file, CHUNK_SIZE=1_000_000, TOTAL_ROWS = 24_335_559)



# GENERATE IMAGE FROM MODEL
def offset_name(dy, dx):
    if dy == 0 and dx == 0:
        return "centro"

    dy_name = f"m{abs(dy)}" if dy < 0 else f"p{dy}"
    dx_name = f"m{abs(dx)}" if dx < 0 else f"p{dx}"
    return f"{dy_name}_{dx_name}"


def get_shift(channel, y0, y1, x0, x1):
    return channel[y0:y1, x0:x1].reshape(-1)


def build_patch_dataframe(img_rgb, radius):
    alto, ancho, _ = img_rgb.shape
    pad = radius

    img_pad = np.pad(
        img_rgb,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    ).astype(np.float32)

    R = img_pad[:, :, 0]
    G = img_pad[:, :, 1]
    B = img_pad[:, :, 2]

    datos = {}

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            name = offset_name(dy, dx)

            y_start = pad + dy
            y_end   = pad + dy + alto
            x_start = pad + dx
            x_end   = pad + dx + ancho

            datos[f'R_{name}'] = get_shift(R, y_start, y_end, x_start, x_end)
            datos[f'G_{name}'] = get_shift(G, y_start, y_end, x_start, x_end)
            datos[f'B_{name}'] = get_shift(B, y_start, y_end, x_start, x_end)

    return pd.DataFrame(datos), alto, ancho


def add_feature_engineering(df, EPS=1e-6):
    rgb_feature_cols = [
        c for c in df.columns
        if c.startswith("R_") or c.startswith("G_") or c.startswith("B_")
    ]

    R_cols = sorted([c for c in rgb_feature_cols if c.startswith("R_")])
    G_cols = sorted([c for c in rgb_feature_cols if c.startswith("G_")])
    B_cols = sorted([c for c in rgb_feature_cols if c.startswith("B_")])

    positions = [c[2:] for c in R_cols]

    R = df['R_centro']
    G = df['G_centro']
    B = df['B_centro']

    sum_rgb = R + G + B + EPS

    df['GLI_centro'] = (2 * G - R - B) / (2 * G + R + B + EPS)
    df['ExG_centro'] = 2 * G - R - B
    df['VARI_centro'] = (G - R) / (G + R - B + EPS)

    df['r_norm_centro'] = R / sum_rgb
    df['g_norm_centro'] = G / sum_rgb
    df['b_norm_centro'] = B / sum_rgb

    for pos in positions:
        Rn = df[f'R_{pos}']
        Gn = df[f'G_{pos}']
        Bn = df[f'B_{pos}']

        df[f'GLI_{pos}'] = (2 * Gn - Rn - Bn) / (2 * Gn + Rn + Bn + EPS)
        df[f'ExG_{pos}'] = 2 * Gn - Rn - Bn

    df['R_mean_patch'] = df[R_cols].mean(axis=1)
    df['G_mean_patch'] = df[G_cols].mean(axis=1)
    df['B_mean_patch'] = df[B_cols].mean(axis=1)

    df['R_std_patch'] = df[R_cols].std(axis=1)
    df['G_std_patch'] = df[G_cols].std(axis=1)
    df['B_std_patch'] = df[B_cols].std(axis=1)

    df['R_min_patch'] = df[R_cols].min(axis=1)
    df['G_min_patch'] = df[G_cols].min(axis=1)
    df['B_min_patch'] = df[B_cols].min(axis=1)

    df['R_max_patch'] = df[R_cols].max(axis=1)
    df['G_max_patch'] = df[G_cols].max(axis=1)
    df['B_max_patch'] = df[B_cols].max(axis=1)

    GLI_cols = [f'GLI_{pos}' for pos in positions]
    ExG_cols = [f'ExG_{pos}' for pos in positions]

    df['GLI_mean_patch'] = df[GLI_cols].mean(axis=1)
    df['GLI_std_patch'] = df[GLI_cols].std(axis=1)

    df['ExG_mean_patch'] = df[ExG_cols].mean(axis=1)
    df['ExG_std_patch'] = df[ExG_cols].std(axis=1)

    # ---------- Contraste local ----------
    df['R_center_minus_mean'] = df['R_centro'] - df['R_mean_patch']
    df['G_center_minus_mean'] = df['G_centro'] - df['G_mean_patch']
    df['B_center_minus_mean'] = df['B_centro'] - df['B_mean_patch']

    df['GLI_center_minus_mean'] = df['GLI_centro'] - df['GLI_mean_patch']
    df['ExG_center_minus_mean'] = df['ExG_centro'] - df['ExG_mean_patch']

    # ---------- Ratios útiles del centro ----------
    df['G_R_ratio_centro'] = df['G_centro'] / (df['R_centro'] + EPS)
    df['G_B_ratio_centro'] = df['G_centro'] / (df['B_centro'] + EPS)
    df['R_B_ratio_centro'] = df['R_centro'] / (df['B_centro'] + EPS)

    return df


def generate_tif_image(img_transform, model_path, PATCH_RADIUS=2, CHUNK_SIZE=500_000):
    modelo = joblib.load(model_path)
    if not hasattr(modelo, "feature_names_in_"):
        raise ValueError(
            "This model doesn't save feature_names_in_. "
            "You need a model trained with modern pandas/sklearn."
        )

    expected_columns = list(modelo.feature_names_in_)
    print(f"Model Loaded. Number of expected features: {len(expected_columns)}")

    img_transform = cv2.cvtColor(img_transform, cv2.COLOR_BGR2RGB)
    alto, ancho, canales = img_transform.shape
    print("Extracting 5x5 patch...")
    df_inferencia, alto, ancho = build_patch_dataframe(img_transform, PATCH_RADIUS)

    print("Calculating feature engineering...")
    df_inferencia = add_feature_engineering(df_inferencia)

    missing = [c for c in expected_columns if c not in df_inferencia.columns]
    extra = [c for c in df_inferencia.columns if c not in expected_columns]

    if missing:
        raise ValueError(f"Columns missing for the model: {missing[:20]}")

    if extra:
        print(f"There are extra calculated columns that the model will not use: {len(extra)}")

    df_inferencia = df_inferencia[expected_columns].astype(np.float32)
    predicciones = []

    for i in tqdm(range(0, len(df_inferencia), CHUNK_SIZE), desc="Predicting"):
        chunk = df_inferencia.iloc[i:i + CHUNK_SIZE]
        pred_chunk = modelo.predict(chunk)
        predicciones.append(pred_chunk)

    predicciones = np.concatenate(predicciones, axis=0)
    print("Rebuilding the final image...")
    img_nir_sintetico = predicciones.reshape((alto, ancho))
    img_nir_sintetico = np.clip(img_nir_sintetico, 0, 255).astype(np.uint8)
    cv2.imshow("synth", img_nir_sintetico)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def calculate_ndvi_map(nir_array, red_array):
    """
    Calculates the NDVI map and returns the array along with a dictionary of metrics.
    
    Args:
        nir_array (np.ndarray): NIR channel array.
        red_array (np.ndarray): Red channel array.
        
    Returns:
        tuple: (ndvi_map, metrics_dict)
    """
    # Ensure images are single channel
    if nir_array.ndim == 3:
        nir_array = nir_array[:, :, 0]
    if red_array.ndim == 3:
        red_array = red_array[:, :, 0]

    # Convert to float for calculation
    nir = nir_array.astype(np.float32)
    red = red_array.astype(np.float32)

    # Calculate NDVI using epsilon to avoid division by zero
    epsilon = 1e-8
    denominator = (nir + red) + epsilon
    ndvi_map = (nir - red) / denominator
    
    # Clip values to the scientific range [-1, 1]
    ndvi_map = np.clip(ndvi_map, -1.0, 1.0)

    # Generate metrics dictionary
    metrics = {
        "min": float(np.min(ndvi_map)),
        "max": float(np.max(ndvi_map)),
        "mean": float(np.mean(ndvi_map)),
        "std": float(np.std(ndvi_map)),
        "median": float(np.median(ndvi_map))
    }

    return ndvi_map, metrics


#model_path = "../data/models/R_random_forest_5x5_features_50p_v1.joblib"
#IMAGE_IN_PATH = "../data/training/test_images/test_red.jpg"
#img_transform = cv2.imread(IMAGE_IN_PATH, cv2.IMREAD_COLOR)
#generate_tif_image(img_transform, model_path, PATCH_RADIUS=2, CHUNK_SIZE=500_000)


