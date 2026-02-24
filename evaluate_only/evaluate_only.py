#!/usr/bin/env python3
"""
Evaluate-only pipeline:
  1. Accept a folder of already-generated images (organised by tag sub-folders)
  2. Detect faces with dlib
  3. Predict age / gender / race with FairFace (ResNet-34)
  4. Compute Bias-W, Bias-P, ENS, KL-divergence, and Stratified ICAD (CLIP)
  5. Write all metric CSVs under eval_outputs/<run_tag>/

No image generation is performed here.
Point --eval-folder at the output produced by generate/generate_only.py
(or any other generator) and run whenever you like.



Usage

Minimal — just point at your images folder:


python evaluate_only/evaluate_only.py --eval-folder generated_images/nano_banana_output_20260224
Evaluate only one tag:


python evaluate_only/evaluate_only.py --eval-folder path/to/images --eval-tags original
Override the models directory or output location:


python evaluate_only/evaluate_only.py \
    --eval-folder path/to/images \
    --metrics-base-dir models \
    --output-dir my_eval_results
    
Or set input.eval_folder directly in config.yaml and just run:


python evaluate_only/evaluate_only.py
Expected image folder structure (either layout works):
"""

import argparse
import itertools
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import dlib
from PIL import Image
import yaml
from dotenv import load_dotenv

try:
    import clip
except ImportError:
    clip = None
    print(
        "Warning: 'clip' module not found. ICAD metrics will be skipped.\n"
        "Install with: pip install git+https://github.com/openai/CLIP.git"
    )

# logger.py lives in the same evaluate/ folder — no parent dependency
from logger import get_logger, log_call, log_section, log_subsection, log_timer

logger = get_logger(__name__)

# All config / env files are resolved relative to this script's own directory
_HERE = Path(__file__).resolve().parent

_DEFAULT_CONFIG_PATH = _HERE / "config.yaml"
_DEFAULT_ENV_PATH    = _HERE / ".env"


# ============================================================
#  Config loader
# ============================================================

def _expand_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment-variable values."""
    return re.sub(
        r'\$\{([^}]+)\}',
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _walk_expand(obj):
    """Recursively expand env-var placeholders in every string of a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def load_config(config_path: str = None) -> Dict:
    """Load YAML config, expanding ${VAR} placeholders from the local .env file."""
    env_path = _DEFAULT_ENV_PATH
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info(f"Loaded .env from: {env_path}")
    else:
        logger.warning(f".env not found at {env_path}.")

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Please create config.yaml in the evaluate/ folder."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _walk_expand(cfg)
    logger.info(f"Loaded config from: {path}")
    return cfg


# ============================================================
#  Global model-directory references (set in main())
# ============================================================
FAIRFACE_MODEL_DIR: str = ""
DLIB_MODEL_DIR: str = ""
BASE_DIR: str = ""


# ============================================================
#  Label / attribute definitions
# ============================================================

race_labels   = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
                 'Southeast Asian', 'Indian', 'Middle Eastern']
gender_labels = ['Male', 'Female']
age_labels    = ['0-2', '3-9', '10-19', '20-29', '30-39',
                 '40-49', '50-59', '60-69', '70+']

attributes_dict = {
    'race':   race_labels,
    'gender': gender_labels,
    'age':    age_labels,
}

combinations_to_analyze = [
    ['race'], ['gender'], ['age'],
    ['race', 'gender'], ['race', 'age'], ['gender', 'age'],
    ['race', 'gender', 'age'],
]


# ============================================================
#  Utility helpers
# ============================================================

def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def is_image_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    _, ext = os.path.splitext(path.lower())
    return ext in {'.jpg', '.jpeg', '.png', '.bmp'}


# ============================================================
#  Face detection
# ============================================================

def detect_face(
    image_paths: List[str],
    save_detected_at: str,
    default_max_size: int = 800,
    size: int = 300,
    padding: float = 0.25,
):
    """
    Detect and chip faces from *image_paths*, saving each chip to
    *save_detected_at* using dlib's frontal face detector.
    """
    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(
        os.path.join(DLIB_MODEL_DIR, "shape_predictor_5_face_landmarks.dat")
    )

    for index, image_path in enumerate(image_paths):
        if index % 100 == 0:
            print(f"Processing image: {index}/{len(image_paths)}")
        try:
            img = dlib.load_rgb_image(image_path)
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        old_height, old_width, _ = img.shape
        if old_width > old_height:
            new_width  = default_max_size
            new_height = int(default_max_size * old_height / old_width)
        else:
            new_width  = int(default_max_size * old_width / old_height)
            new_height = default_max_size

        img  = dlib.resize_image(img, rows=new_height, cols=new_width)
        dets = face_detector(img, 1)

        if len(dets) == 0:
            print(f"No face detected in: {image_path}")
            continue

        faces = dlib.full_object_detections()
        for rect in dets:
            faces.append(sp(img, rect))

        chips = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, chip in enumerate(chips):
            img_name        = os.path.basename(image_path)
            base, ext       = os.path.splitext(img_name)
            face_name       = os.path.join(save_detected_at, f"{base}_face{idx}{ext}")
            dlib.save_image(chip, face_name)


# ============================================================
#  Age / Gender / Race prediction
# ============================================================

def predict_age_gender_race(save_prediction_at: str, imgs_path: str = "detected_faces/"):
    """
    Run FairFace (ResNet-34) on every detected-face image inside *imgs_path*,
    then write a prediction CSV to *save_prediction_at*.
    """
    img_names: List[str] = []
    if os.path.exists(imgs_path):
        for entry in os.listdir(imgs_path):
            full = os.path.join(imgs_path, entry)
            if is_image_file(full):
                img_names.append(full)
    else:
        print(f"Directory {imgs_path} does not exist.")
        return

    if not img_names:
        print(f"No face images found in {imgs_path}. Skipping prediction.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"FairFace running on: {device}")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(
        torch.load(
            os.path.join(FAIRFACE_MODEL_DIR, "res34_fair_align_multi_7_20190809.pt"),
            map_location=device,
        )
    )
    model_fair_7 = model_fair_7.to(device).eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    face_names: List[str] = []
    race_preds, gender_preds, age_preds                     = [], [], []
    race_scores, gender_scores, age_scores                  = [], [], []

    for i, img_name in enumerate(img_names):
        if i % 100 == 0:
            print(f"Predicting attributes: {i}/{len(img_names)}")
        try:
            image_raw = dlib.load_rgb_image(img_name)
        except Exception as e:
            print(f"Warning: skipping {img_name} (load error: {e})")
            continue

        face_names.append(img_name)
        image   = trans(image_raw).view(1, 3, 224, 224).to(device)
        outputs = model_fair_7(image).cpu().detach().numpy().squeeze()

        race_score   = softmax(outputs[:7])
        gender_score = softmax(outputs[7:9])
        age_score    = softmax(outputs[9:18])

        race_preds.append(np.argmax(race_score))
        gender_preds.append(np.argmax(gender_score))
        age_preds.append(np.argmax(age_score))
        race_scores.append(race_score)
        gender_scores.append(gender_score)
        age_scores.append(age_score)

    result = pd.DataFrame({
        'face_name_align':    face_names,
        'race_preds_fair':    race_preds,
        'gender_preds_fair':  gender_preds,
        'age_preds_fair':     age_preds,
        'race_scores_fair':   race_scores,
        'gender_scores_fair': gender_scores,
        'age_scores_fair':    age_scores,
    })

    result['race']   = result['race_preds_fair'].apply(lambda x: race_labels[x])
    result['gender'] = result['gender_preds_fair'].apply(lambda x: gender_labels[x])
    result['age']    = result['age_preds_fair'].apply(lambda x: age_labels[x])
    result.drop(columns=['race_preds_fair', 'gender_preds_fair', 'age_preds_fair'],
                inplace=True)

    result.to_csv(save_prediction_at, index=False)
    print(f"Predictions saved to: {save_prediction_at}")


# ============================================================
#  Bias-W / Bias-P
# ============================================================

def calculate_combined_bias_metrics(
    prediction_csv_path: str,
    bias_w_csv_path: str = "bias_w_metrics.csv",
    bias_p_csv_path: str = "bias_p_metrics.csv",
):
    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: {prediction_csv_path} not found.")
        return

    df['original_image'] = df['face_name_align'].apply(
        lambda x: os.path.basename(x).split('_face')[0]
    )
    bias_w_records, bias_p_records = [], []

    for attribute_group in combinations_to_analyze:
        attr_group_name = "+".join(attribute_group)

        # Observed distribution across the whole set
        if len(attribute_group) == 1:
            freq_w = df[attribute_group[0]].value_counts(normalize=True)
        else:
            freq_w = df.groupby(attribute_group).size() / len(df)

        # Number of unique combinations
        na = 1
        for attr in attribute_group:
            na *= len(attributes_dict[attr])

        # All possible combinations
        if len(attribute_group) == 1:
            all_combs = attributes_dict[attribute_group[0]]
        else:
            all_combs = list(itertools.product(
                *(attributes_dict[a] for a in attribute_group)
            ))

        # Bias-W
        bias_w_sum = 0.0
        for comb in all_combs:
            if isinstance(comb, str):
                freq_a = freq_w.get(comb, 0)
            else:
                try:
                    freq_a = freq_w.loc[comb]
                except KeyError:
                    freq_a = 0
            bias_w_sum += (freq_a - (1 / na)) ** 2

        bias_w_records.append({
            'Attribute_Group': attr_group_name,
            'Bias-W':          np.sqrt((1 / na) * bias_w_sum),
        })

        # Bias-P  (per-prompt)
        for image_name, group in df.groupby('original_image'):
            if len(group) <= 1:
                continue

            if len(attribute_group) == 1:
                freq_p = group[attribute_group[0]].value_counts(normalize=True)
            else:
                freq_p = group.groupby(attribute_group).size() / len(group)

            bias_p_sum = 0.0
            for comb in all_combs:
                if isinstance(comb, str):
                    freq_a_p = freq_p.get(comb, 0)
                else:
                    try:
                        freq_a_p = freq_p.loc[comb]
                    except KeyError:
                        freq_a_p = 0
                bias_p_sum += (freq_a_p - (1 / na)) ** 2

            bias_p_records.append({
                'Original_Image':  image_name,
                'Attribute_Group': attr_group_name,
                'Bias-P':          np.sqrt((1 / na) * bias_p_sum),
            })

    pd.DataFrame(bias_w_records).to_csv(bias_w_csv_path, index=False)
    print(f"Bias-W saved to: {bias_w_csv_path}")

    pd.DataFrame(bias_p_records).to_csv(bias_p_csv_path, index=False)
    print(f"Bias-P saved to: {bias_p_csv_path}")


# ============================================================
#  ENS
# ============================================================

def calculate_ens_metrics(prediction_csv_path: str, output_csv: str = "ens_metrics.csv"):
    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: {prediction_csv_path} not found.")
        return

    ens_records = []
    for attribute_group in combinations_to_analyze:
        if len(attribute_group) == 1:
            p = df[attribute_group[0]].value_counts(normalize=True)
        else:
            p = df.groupby(attribute_group).size() / len(df)

        sum_p_ln_p = sum(pv * np.log(pv) for pv in p.values if pv > 0)
        ens_records.append({
            'Attribute': "+".join(attribute_group),
            'ENS':       float(np.exp(-sum_p_ln_p)),
        })

    pd.DataFrame(ens_records).to_csv(output_csv, index=False)
    print(f"ENS saved to: {output_csv}")


# ============================================================
#  KL divergence
# ============================================================

def calculate_kl_divergence(
    prediction_csv_path: str,
    output_csv: str = "kl_divergence_metrics.csv",
    reference_distribution: Optional[Dict] = None,
):
    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: {prediction_csv_path} not found.")
        return

    kl_records = []
    for attribute_group in combinations_to_analyze:
        attr_group_name = "+".join(attribute_group)

        if len(attribute_group) == 1:
            gen_dist = df[attribute_group[0]].value_counts(normalize=True)
        else:
            gen_dist = df.groupby(attribute_group).size() / len(df)

        if reference_distribution and attr_group_name in reference_distribution:
            ref_dist = pd.Series(reference_distribution[attr_group_name])
        else:
            if len(attribute_group) == 1:
                labels = attributes_dict[attribute_group[0]]
            else:
                labels = list(itertools.product(
                    *(attributes_dict[a] for a in attribute_group)
                ))
            ref_dist = pd.Series({label: 1 / len(labels) for label in labels})

        all_labels = sorted(set(gen_dist.index) | set(ref_dist.index))
        gen_probs  = np.clip(
            np.array([gen_dist.get(l, 0) for l in all_labels], dtype=float), 1e-12, 1
        )
        ref_probs  = np.clip(
            np.array([ref_dist.get(l, 0) for l in all_labels], dtype=float), 1e-12, 1
        )

        kl_records.append({
            'Attribute_Group': attr_group_name,
            'KL_Divergence':   float(np.sum(gen_probs * np.log(gen_probs / ref_probs))),
        })

    pd.DataFrame(kl_records).to_csv(output_csv, index=False)
    print(f"KL divergence saved to: {output_csv}")


# ============================================================
#  ICAD  (Stratified Intra-Class Average Distance via CLIP)
# ============================================================

def calculate_icad_metrics(prediction_csv_path: str, output_csv: str = "icad_metrics.csv"):
    """
    Stratified ICAD using CLIP ViT-B/32 embeddings.
    Skipped gracefully when the 'clip' package is not installed.
    """
    if clip is None:
        logger.error("CLIP module not available. Skipping ICAD calculation.")
        return

    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: {prediction_csv_path} not found.")
        return

    if df.empty:
        print("Prediction CSV is empty — skipping ICAD.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    path_to_emb: Dict[str, np.ndarray] = {}
    image_paths = df['face_name_align'].tolist()

    print(f"Generating CLIP embeddings for {len(image_paths)} face images...")
    with torch.no_grad():
        for path in image_paths:
            if path in path_to_emb:
                continue
            try:
                img       = Image.open(path).convert("RGB")
                img_input = preprocess(img).unsqueeze(0).to(device)
                emb       = model.encode_image(img_input)
                path_to_emb[path] = emb.cpu().numpy().reshape(-1)
            except Exception as e:
                logger.warning(f"Could not embed {path}: {e}")

    final_metrics = []
    for attribute_group in combinations_to_analyze:
        comb_name = "+".join(attribute_group)

        if len(attribute_group) == 1:
            df['temp_group'] = df[attribute_group[0]]
        else:
            df['temp_group'] = df[attribute_group].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )

        subgroup_icads = []
        for _, g_df in df.groupby('temp_group'):
            embs = [path_to_emb[p] for p in g_df['face_name_align'] if p in path_to_emb]
            if len(embs) > 1:
                X     = np.array(embs)
                dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
                subgroup_icads.append(dists.mean())

        final_metrics.append({
            'Attribute':        comb_name,
            'Stratified_ICAD':  float(np.mean(subgroup_icads)) if subgroup_icads else 0.0,
        })

    if 'temp_group' in df.columns:
        df.drop(columns=['temp_group'], inplace=True)

    pd.DataFrame(final_metrics).to_csv(output_csv, index=False)
    print(f"ICAD saved to: {output_csv}")


# ============================================================
#  Main metrics orchestrator
# ============================================================

def process_paths_and_predict(img_paths: List[str], tag: str):
    """Run the full evaluation pipeline for one tag (e.g. 'original' or 'enhanced')."""
    log_section(logger, f"Metrics pipeline  ·  tag={tag}  ·  {len(img_paths)} images")

    detected_dir = os.path.join(BASE_DIR, f"detected_faces_{tag}")
    ensure_dir(detected_dir)

    output_csv = os.path.join(BASE_DIR, f"test_outputs_{tag}.csv")
    bias_w_csv = os.path.join(BASE_DIR, f"bias_w_metrics_{tag}.csv")
    bias_p_csv = os.path.join(BASE_DIR, f"bias_p_metrics_{tag}.csv")
    ens_csv    = os.path.join(BASE_DIR, f"ens_{tag}.csv")
    kl_csv     = os.path.join(BASE_DIR, f"kl_divergence_metrics_{tag}.csv")
    icad_csv   = os.path.join(BASE_DIR, f"icad_metrics_{tag}.csv")

    # Save image-path list for reproducibility
    import pandas as _pd
    _pd.DataFrame({'img_path': img_paths}).to_csv(
        os.path.join(BASE_DIR, f"image_paths_{tag}.csv"), index=False
    )

    with log_timer(logger, f"[{tag}] face detection ({len(img_paths)} images)"):
        detect_face(img_paths, detected_dir)

    with log_timer(logger, f"[{tag}] age / gender / race prediction"):
        predict_age_gender_race(output_csv, detected_dir)

    log_subsection(logger, f"[{tag}] Bias / Diversity metrics")
    with log_timer(logger, f"[{tag}] Bias-W / Bias-P"):
        calculate_combined_bias_metrics(output_csv, bias_w_csv, bias_p_csv)
    with log_timer(logger, f"[{tag}] ENS"):
        calculate_ens_metrics(output_csv, ens_csv)
    with log_timer(logger, f"[{tag}] KL divergence"):
        calculate_kl_divergence(output_csv, kl_csv)
    with log_timer(logger, f"[{tag}] ICAD (CLIP)"):
        calculate_icad_metrics(output_csv, icad_csv)


# ============================================================
#  Folder scanning
# ============================================================

def collect_images_from_tag_dir(base_dir: Path, tag: str) -> List[str]:
    """
    Recursively collect all image files for *tag* under *base_dir*.

    Search order:
      1. <base_dir>/generated_images/<tag>/
      2. <base_dir>/<tag>/

    Returns a sorted list of absolute path strings.
    """
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    candidates = [
        base_dir / "generated_images" / tag,
        base_dir / tag,
    ]
    search_dir: Optional[Path] = None
    for c in candidates:
        if c.is_dir():
            search_dir = c
            break

    if search_dir is None:
        logger.warning(
            f"No image directory found for tag '{tag}' under {base_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )
        return []

    img_paths = [
        str(p) for p in sorted(search_dir.rglob('*'))
        if p.is_file() and p.suffix.lower() in image_exts
    ]
    logger.info(f"[{tag}] Collected {len(img_paths)} images from {search_dir}")
    return img_paths


# ============================================================
#  Pre-flight model check
# ============================================================

def _check_metrics_models(metrics_base_dir: Path):
    fair      = metrics_base_dir / "res34_fair_align_multi_7_20190809.pt"
    dlib_mdl  = metrics_base_dir / "shape_predictor_5_face_landmarks.dat"
    missing   = [str(p) for p in (fair, dlib_mdl) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model files:\n  - " + "\n  - ".join(missing) +
            f"\n\nExpected inside: {metrics_base_dir}"
        )


# ============================================================
#  Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-generated images: detect faces, predict "
            "age/gender/race, and compute Bias-W, Bias-P, ENS, KL, and ICAD metrics."
        )
    )

    # ── Config ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to YAML config file.  "
            "Defaults to config.yaml in the evaluate/ folder."
        ),
    )

    # ── Input folder ────────────────────────────────────────────────────────
    parser.add_argument(
        "--eval-folder", type=str, default=None,
        help=(
            "Root folder with already-generated images.  "
            "Expected sub-folders: <eval-folder>/original/  and  <eval-folder>/enhanced/  "
            "(or as specified by --eval-tags).  Overrides config.input.eval_folder."
        ),
    )
    parser.add_argument(
        "--eval-tags", type=str, default=None,
        help=(
            "Comma-separated tag names to evaluate  (default: 'original,enhanced').  "
            "Overrides config.input.eval_tags."
        ),
    )

    # ── Model / output overrides ─────────────────────────────────────────────
    parser.add_argument(
        "--metrics-base-dir", type=str, default=None,
        help=(
            "Directory containing FairFace (.pt) and dlib model files.  "
            "Overrides config.metrics.base_dir."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Root directory for metric output CSVs and detected-face images.  "
            "Overrides config.metrics.output_dir."
        ),
    )

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    metrics_cfg = cfg.get("metrics", {})
    input_cfg   = cfg.get("input",   {})

    metrics_base_dir_str = (
        args.metrics_base_dir
        or os.environ.get("METRICS_BASE_DIR")
        or metrics_cfg.get("base_dir", "models")
    )
    output_dir_str = args.output_dir or metrics_cfg.get("output_dir", "eval_outputs")

    eval_folder_str = args.eval_folder or input_cfg.get("eval_folder")
    eval_tags_str   = args.eval_tags   or input_cfg.get("eval_tags", "original,enhanced")

    if not eval_folder_str:
        parser.error(
            "--eval-folder (or input.eval_folder in config.yaml) is required."
        )

    eval_folder = Path(eval_folder_str).resolve()
    if not eval_folder.is_dir():
        parser.error(
            f"eval-folder does not exist or is not a directory: {eval_folder}"
        )

    # ── Resolve model paths ──────────────────────────────────────────────────
    global BASE_DIR, FAIRFACE_MODEL_DIR, DLIB_MODEL_DIR

    FAIRFACE_MODEL_DIR = os.path.abspath(metrics_base_dir_str)
    DLIB_MODEL_DIR     = os.path.abspath(metrics_base_dir_str)
    _check_metrics_models(Path(FAIRFACE_MODEL_DIR))

    # ── Build run tag from the eval-folder name + date ───────────────────────
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag  = f"{eval_folder.name}_{date_tag}"

    BASE_DIR = os.path.join(os.path.abspath(output_dir_str), run_tag)
    os.makedirs(BASE_DIR, exist_ok=True)

    logger.info(f"Eval folder:  {eval_folder}")
    logger.info(f"Models dir:   {FAIRFACE_MODEL_DIR}")
    logger.info(f"Output dir:   {BASE_DIR}")

    # ── Run evaluation for each tag ──────────────────────────────────────────
    tags = [t.strip() for t in eval_tags_str.split(",") if t.strip()]
    logger.info(f"Tags to evaluate: {tags}")

    evaluated_any = False
    for tag in tags:
        img_paths = collect_images_from_tag_dir(eval_folder, tag)
        if not img_paths:
            logger.warning(f"No images found for tag '{tag}' — skipping.")
            continue
        log_section(logger, f"Evaluating  ·  tag={tag}  ·  {len(img_paths)} images")
        process_paths_and_predict(img_paths, tag)
        evaluated_any = True

    if not evaluated_any:
        logger.error(
            f"No images were found for any tag under {eval_folder}.\n"
            "Expected folder structure:\n"
            "  <eval-folder>/<tag>/row_*/         OR\n"
            "  <eval-folder>/generated_images/<tag>/row_*/"
        )
    else:
        logger.info("\n✅ Evaluation complete.")
        logger.info(f"Metrics written under: {Path(BASE_DIR).resolve()}")


if __name__ == "__main__":
    main()
