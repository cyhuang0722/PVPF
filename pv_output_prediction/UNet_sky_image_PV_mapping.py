import os
import ast
import datetime as dt
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

import json
import logging
import sys
import platform


# ###############################################################################
# # Configuration
# ###############################################################################

DERIVED_DIR = Path("./derived")
DERIVED_PARQUET = DERIVED_DIR / "pv_windows_simple.parquet"
DERIVED_CSV = DERIVED_DIR / "pv_windows_simple.csv"

MODEL_OUTPUT_DIR = Path("./model_output")
PACK_DIR = DERIVED_DIR / "packed"

TEST_DATE = dt.date(2026, 2, 23)  # 最后一天留作测试集

NUM_EPOCHS = 200
NUM_FOLDS = 5
BATCH_SIZE = 128
BASE_LR = 1e-4
NUM_FILTERS = 12
RANDOM_SEED = 42

# 归一化用的 PV 峰值（W）
PEAK_POWER_W = 66.3 * 1000.0

# 图像尺寸（统一 resize），通道数按 3 通道彩色图像处理
# 注意要与 preproces_data.CFG.IMG_HEIGHT / IMG_WIDTH 保持一致
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CHANNELS = 3

# Sky mask path for scheme B
SKY_MASK_PATH = Path("./sky_mask.png")

# 可选：训练时最多使用多少个 train+val 样本（None 表示全量）
MAX_TRAINVAL_SAMPLES: int | None = None
SHUFFLE_BEFORE_SPLIT: bool = True  # 小数据下让 train/val 分布更接近，减少 val_loss 抖动



###############################################################################
# Reproducible run directory + logging
###############################################################################


def make_run_dir(base_dir: Path) -> Path:
    """Create a timestamped run directory for traceable archives."""
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    """Log to both console and a file under run_dir."""
    log_path = run_dir / "train.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers (important in notebooks / repeated runs)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    logging.info(f"[Logging] Writing logs to: {log_path}")


def dump_run_metadata(run_dir: Path, cfg: dict) -> None:
    """Persist configuration + environment metadata for traceability."""
    meta = {
        "timestamp": dt.datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "tensorflow": getattr(tf, "__version__", "unknown"),
        "keras": getattr(keras, "__version__", "unknown"),
        "cfg": cfg,
    }
    out = run_dir / "run_metadata.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"[Archive] Saved run metadata: {out}")


###############################################################################
# Device configuration (CPU / GPU)
###############################################################################


def configure_devices() -> None:
    """Configure TensorFlow to use GPU if available, otherwise CPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logging.info(f"[Device] Detected {len(gpus)} GPU(s): {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            logging.info(f"[Device] Using logical GPU(s): {logical_gpus}")
        except Exception as e:
            logging.warning(f"[Device] Failed to set memory growth, will still use GPU: {e}")
    else:
        logging.info("[Device] No GPU detected, using CPU.")



###############################################################################
# Data loading & splitting
###############################################################################


# Helper: load and resize sky mask for scheme B
def load_and_resize_sky_mask(mask_path: Path, out_h: int, out_w: int) -> np.ndarray:
    """Load a binary-ish sky mask image and resize to model input size.

    Returns:
      mask: (out_h, out_w, 1) float32 in {0.0, 1.0}
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Sky mask not found: {mask_path}")

    # Load as grayscale
    m = Image.open(mask_path).convert("L")
    # Resize with nearest-neighbor to preserve hard edges
    m = m.resize((out_w, out_h), resample=Image.NEAREST)
    m = np.asarray(m, dtype=np.float32) / 255.0
    # Binarize robustly (in case mask isn't perfectly 0/255)
    m = (m >= 0.5).astype(np.float32)
    return m[..., None]


def load_packed_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-packed image batches (.npz) produced by preproces_data.pack_windows_to_npz.

    Returns:
      images: (N, T, H, W, C) float32 in [0,1]
      pv:     (N,) float32
      times:  (N,) array of timestamps (as strings)
    """
    if not PACK_DIR.exists():
        raise FileNotFoundError(
            f"PACK_DIR {PACK_DIR} 不存在，请先在 preproces_data.cfg 中开启 ENABLE_PACKING "
            f"并运行 preproces_data.py 生成 packed 数据。"
        )

    shard_paths = sorted(PACK_DIR.glob("batch_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(
            f"在 {PACK_DIR} 下找不到 batch_*.npz 文件，请确认预打包步骤已完成。"
        )

    images_list: list[np.ndarray] = []
    pv_list: list[np.ndarray] = []
    ts_list: list[np.ndarray] = []

    logging.info(f"[load_packed_arrays] Found {len(shard_paths)} shards in {PACK_DIR}")
    for p in shard_paths:
        data = np.load(p, allow_pickle=True)
        images_list.append(data["images"])       # (B,T,H,W,C)
        pv_list.append(data["power"])            # (B,)
        ts_list.append(data["ts_power_end"])     # (B,)

    images = np.concatenate(images_list, axis=0)
    pv = np.concatenate(pv_list, axis=0).astype(np.float32)
    times = np.concatenate(ts_list, axis=0)

    logging.info(
        f"[load_packed_arrays] Loaded images shape {images.shape}, "
        f"pv shape {pv.shape}, times shape {times.shape}"
    )
    return images, pv, times


def to_date_array(times: np.ndarray) -> np.ndarray:
    """Convert generic timestamp array to date-only numpy array."""
    dates: List[dt.date] = []
    for t in times:
        if isinstance(t, dt.datetime):
            dates.append(t.date())
        elif isinstance(t, dt.date):
            dates.append(t)
        else:
            # 字符串或 numpy.datetime64：统一走 pandas 解析，避免 tz 警告
            ts = pd.to_datetime(t)
            dates.append(ts.date())
    return np.array(dates)


def split_trainval_test(indices: np.ndarray, dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/val pool and test set based on TEST_DATE.

    If no samples exist for TEST_DATE, fallback to using the last available date
    in `dates` as the test day, and log a warning.
    """
    test_mask = dates == TEST_DATE
    idx_test = indices[test_mask]
    idx_trainval = indices[~test_mask]

    if len(idx_test) == 0:
        # fallback: use the last available date as test day
        last_date = np.max(dates)
        logging.warning(
            f"[Warning] 在日期 {TEST_DATE} 找不到测试样本，"
            f"自动改用数据中的最后一天 {last_date} 作为测试集。"
        )
        test_mask = dates == last_date
        idx_test = indices[test_mask]
        idx_trainval = indices[~test_mask]
        if len(idx_test) == 0:
            raise ValueError(
                f"在日期 {TEST_DATE} 或最后一天 {last_date} 都找不到测试样本，请检查时间戳。"
            )

    return idx_trainval, idx_test


###############################################################################
# Data generator: use pre-packed images (N,T,H,W,C) -> (H,W,T*C)
###############################################################################


class WindowSequence(keras.utils.Sequence):
    def __init__(
        self,
        images: np.ndarray,  # (N,T,H,W,C)
        pv: np.ndarray,      # (N,)
        indices: np.ndarray,
        batch_size: int,
        sky_mask: np.ndarray | None = None,  # (H,W,1) float32 in {0,1}
        shuffle: bool = True,
    ):
        self.images = images
        self.pv = pv
        self.indices = np.array(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Keep one RNG instance; do NOT re-seed every epoch
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.on_epoch_end()

        self.sky_mask = sky_mask
        if self.sky_mask is not None:
            # Basic sanity checks
            if self.sky_mask.ndim != 3 or self.sky_mask.shape[-1] != 1:
                raise ValueError(f"sky_mask must have shape (H,W,1), got {self.sky_mask.shape}")

        # cache shapes
        _, self.time_steps, self.img_h, self.img_w, self.num_channels = images.shape
        self.total_channels = self.time_steps * self.num_channels
        self.total_channels_with_mask = self.total_channels + (1 if self.sky_mask is not None else 0)

        if self.sky_mask is not None:
            if (self.sky_mask.shape[0], self.sky_mask.shape[1]) != (self.img_h, self.img_w):
                raise ValueError(
                    f"sky_mask shape {self.sky_mask.shape} does not match packed image size "
                    f"({self.img_h},{self.img_w},1). Did you resize it to IMG_HEIGHT/IMG_WIDTH?"
                )

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_size = len(batch_indices)

        # slice pre-packed images (B,T,H,W,C)
        imgs = self.images[batch_indices]  # (B,T,H,W,C)
        # Apply sky mask to remove buildings/ground content (scheme B)
        if self.sky_mask is not None:
            imgs = imgs * self.sky_mask[None, None, ...]  # broadcast to (B,T,H,W,C)
        # reshape to (B,H,W,T*C)
        imgs = np.transpose(imgs, (0, 2, 3, 1, 4))  # (B,H,W,T,C)
        imgs = imgs.reshape(batch_size, self.img_h, self.img_w, self.total_channels)
        if self.sky_mask is not None:
            mask_batch = np.repeat(self.sky_mask[None, ...], repeats=batch_size, axis=0)  # (B,H,W,1)
            imgs = np.concatenate([imgs, mask_batch], axis=-1)  # (B,H,W,T*C+1)

        y_batch = self.pv[batch_indices].astype(np.float32)
        return imgs, y_batch

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 1:
            self.rng.shuffle(self.indices)


###############################################################################
# Model: Image2PV (参考 notebook, 输入改为 T*C 通道)
###############################################################################


def build_image2pv_model(image_input_dim, num_filters: int = NUM_FILTERS) -> keras.Model:
    x_in = keras.Input(shape=image_input_dim)

    def conv3x3_block(x, channels: int):
        x = keras.layers.Conv2D(filters=channels, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x

    def bottleneck_block(x, channels: int):
        residual = x
        x = keras.layers.Conv2D(filters=channels, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(filters=channels, kernel_size=(3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([residual, x])
        return x

    def up2x2_conv3x3(x, channels: int):
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters=channels, kernel_size=(3, 3), padding="same")(x)
        return x

    # encoder
    x = keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), padding="same")(x_in)
    x1 = conv3x3_block(x, num_filters)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x1)
    x2 = conv3x3_block(x, 2 * num_filters)

    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x2)
    x = conv3x3_block(x, 4 * num_filters)

    # bottleneck residual blocks
    x = bottleneck_block(x, 4 * num_filters)
    x = bottleneck_block(x, 4 * num_filters)

    # decoder
    x = up2x2_conv3x3(x, 2 * num_filters)
    x = keras.layers.Concatenate(axis=3)([x, x2])
    x = conv3x3_block(x, 2 * num_filters)
    x = keras.layers.Dropout(0.4)(x)

    x = up2x2_conv3x3(x, num_filters)
    x = keras.layers.Concatenate(axis=3)([x, x1])
    x = conv3x3_block(x, num_filters)
    x = keras.layers.Dropout(0.4)(x)

    # output head: spatial features -> global pooling -> scalar (more stable than Flatten on small data)
    x = keras.layers.Conv2D(filters=max(8, num_filters), kernel_size=(1, 1), padding="same")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    y_out = keras.layers.Dense(units=1)(x)

    model = keras.Model(inputs=x_in, outputs=y_out, name="Image2PV")
    return model

###############################################################################
# Minimal end-to-end training & evaluation (no K-fold, small config)
###############################################################################


def train_and_evaluate():
    # Ensure base output dir exists and create a per-run archive directory
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(MODEL_OUTPUT_DIR)
    setup_logging(run_dir)

    cfg = {
        "TEST_DATE": str(TEST_DATE),
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_FOLDS": NUM_FOLDS,
        "BATCH_SIZE": BATCH_SIZE,
        "BASE_LR": BASE_LR,
        "NUM_FILTERS": NUM_FILTERS,
        "RANDOM_SEED": RANDOM_SEED,
        "PEAK_POWER_W": PEAK_POWER_W,
        "IMG_HEIGHT": IMG_HEIGHT,
        "IMG_WIDTH": IMG_WIDTH,
        "NUM_CHANNELS": NUM_CHANNELS,
        "SKY_MASK_PATH": str(SKY_MASK_PATH),
        "PACK_DIR": str(PACK_DIR),
        "MAX_TRAINVAL_SAMPLES": MAX_TRAINVAL_SAMPLES,
        "SHUFFLE_BEFORE_SPLIT": SHUFFLE_BEFORE_SPLIT,
    }
    dump_run_metadata(run_dir, cfg)

    # Configure devices on HPC node (GPU if available, else CPU)
    configure_devices()

    # Load pre-packed data (.npz shards)
    images, pv_raw, times = load_packed_arrays()  # images: (N,T,H,W,C), pv_raw in W
    num_samples, time_steps, img_h, img_w, num_channels = images.shape

    if time_steps != 15:
        logging.warning(f"Warning: 期望窗口长度为 15，但数据为 {time_steps}，将按实际值使用。")
    if (img_h, img_w) != (IMG_HEIGHT, IMG_WIDTH) or num_channels != NUM_CHANNELS:
        logging.warning(
            f"Warning: packed image shape {(img_h, img_w, num_channels)} "
            f"与当前配置 {(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)} 不一致。"
        )

    # Normalize power by peak value for training
    pv_norm = pv_raw.astype(np.float32) / PEAK_POWER_W

    # Load & resize sky mask (scheme B)
    sky_mask = load_and_resize_sky_mask(SKY_MASK_PATH, IMG_HEIGHT, IMG_WIDTH)
    logging.info(f"[sky_mask] Loaded {SKY_MASK_PATH} -> resized mask shape {sky_mask.shape}")

    # For now, use all current data for train/val only (no test split).
    all_indices = np.arange(num_samples)
    idx_all = all_indices

    # Optionally subsample for a fast minimal run
    if MAX_TRAINVAL_SAMPLES is not None and len(idx_all) > MAX_TRAINVAL_SAMPLES:
        idx_all = idx_all[:MAX_TRAINVAL_SAMPLES]

    image_input_dim = (IMG_HEIGHT, IMG_WIDTH, time_steps * NUM_CHANNELS + 1)

    logging.info(
        f"Total samples: {num_samples}, "
        f"train+val (after cap): {len(idx_all)}"
    )

    # Simple train/val split on all data (80/20)
    # For small datasets, shuffling before split usually stabilizes val_loss.
    split_idx_all = np.array(idx_all, copy=True)
    if SHUFFLE_BEFORE_SPLIT:
        rng_split = np.random.default_rng(RANDOM_SEED)
        rng_split.shuffle(split_idx_all)

    n_all = len(split_idx_all)
    n_val = max(1, int(0.2 * n_all))
    train_idx = split_idx_all[:-n_val]
    val_idx = split_idx_all[-n_val:]

    train_seq = WindowSequence(
        images=images,
        pv=pv_norm,
        indices=train_idx,
        batch_size=BATCH_SIZE,
        sky_mask=sky_mask,
        shuffle=True,
    )
    val_seq = WindowSequence(
        images=images,
        pv=pv_norm,
        indices=val_idx,
        batch_size=BATCH_SIZE,
        sky_mask=sky_mask,
        shuffle=False,
    )
    # Build and train model (small UNet)
    keras.backend.clear_session()
    model = build_image2pv_model(image_input_dim)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=BASE_LR, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )

    best_model_path = run_dir / "best_model.keras"
    # Keras 3 requires `.weights.h5` when save_weights_only=True
    best_weights_path = run_dir / "best_weights.weights.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_seq,
        epochs=NUM_EPOCHS,
        validation_data=val_seq,
        callbacks=callbacks,
        verbose=2,
    )

    # Load best model for consistent inference
    if best_model_path.exists():
        logging.info(f"[Infer] Loading best model from {best_model_path}")
        best_model = keras.models.load_model(str(best_model_path))
    else:
        logging.warning("[Infer] best_model.keras not found; using in-memory model for inference")
        best_model = model

    # Build inference sequence over ALL used samples (idx_all)
    infer_seq = WindowSequence(
        images=images,
        pv=pv_norm,
        indices=idx_all,
        batch_size=BATCH_SIZE,
        sky_mask=sky_mask,
        shuffle=False,
    )

    y_pred_norm = best_model.predict(infer_seq, verbose=0).reshape(-1).astype(np.float32)
    y_true_norm = pv_norm[idx_all].astype(np.float32)

    # Denormalize to W
    y_pred_w = y_pred_norm * PEAK_POWER_W
    y_true_w = y_true_norm * PEAK_POWER_W

    # Tag split for traceability
    split = np.empty(len(idx_all), dtype=object)
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    for i, gi in enumerate(idx_all.tolist()):
        if gi in train_set:
            split[i] = "train"
        elif gi in val_set:
            split[i] = "val"
        else:
            split[i] = "unknown"

    # Timestamps
    ts_all = times[idx_all]
    ts_parsed = pd.to_datetime(ts_all)
    ts_series = pd.Series(ts_parsed)

    pred_df = pd.DataFrame(
        {
            "ts_power_end": ts_series.astype(str),
            "date": ts_series.dt.date.astype(str),
            "split": split,
            "pv_true_norm": y_true_norm,
            "pv_pred_norm": y_pred_norm,
            "pv_true_W": y_true_w,
            "pv_pred_W": y_pred_w,
        }
    )

    out_pred_csv = run_dir / "predictions_all.csv"
    pred_df.to_csv(out_pred_csv, index=False)
    logging.info(f"[Archive] Saved predictions CSV: {out_pred_csv} (rows={len(pred_df)})")

    val_loss, val_mae = model.evaluate(val_seq, verbose=0)
    # 模型在归一化后的标签上训练：loss/mae 是相对值
    val_loss_w = val_loss * (PEAK_POWER_W ** 2)
    val_mae_w = val_mae * PEAK_POWER_W
    logging.info(
        f"[Minimal] Val MSE={val_loss:.4f} (norm), MAE={val_mae:.4f} (norm); "
        f"MSE={val_loss_w:.2f} W^2, MAE={val_mae_w:.2f} W"
    )

    # Save final in-memory model too (best is saved by checkpoint)
    model.save(str(run_dir / "final_model.keras"))
    model.save_weights(str(run_dir / "final_weights.weights.h5"))
    np.savez(run_dir / "history.npz", **history.history)

    metrics_df = pd.DataFrame(
        [
            {
                "val_loss_norm": float(val_loss),
                "val_mae_norm": float(val_mae),
                "val_loss_W2": float(val_loss_w),
                "val_mae_W": float(val_mae_w),
            }
        ]
    )
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    logging.info(f"[Archive] Saved metrics: {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    train_and_evaluate()
