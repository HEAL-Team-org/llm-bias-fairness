#!/usr/bin/env python3
"""
Concatenated pipeline:
1) Read a CSV of ORIGINAL prompts -> generate images per prompt via a local server
2) Compute bias/diversity metrics (Bias-W, Bias-P, ENS, KL, and ICAD)
3) Read a CSV of ENHANCED prompts -> generate images per prompt via a local server
4) Compute the same metrics for the enhanced set
"""

import argparse
import csv
from datetime import datetime
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
import requests
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import dlib
import itertools

# --- NEW IMPORTS FOR ICAD (CLIP) ---
from PIL import Image
try:
    import clip
except ImportError:
    print("Warning: 'clip' module not found. Please run: pip install git+https://github.com/openai/CLIP.git")
# -----------------------------------

import re
import yaml
from dotenv import load_dotenv

# ----------------------------
# Logging pipeline
# ----------------------------
from logger import get_logger, log_call, log_section, log_subsection, log_timer

logger = get_logger(__name__)


# ----------------------------
# Config loader
# ----------------------------
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_DEFAULT_ENV_PATH    = Path(__file__).parent / ".env"


def _expand_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with their environment-variable values."""
    return re.sub(
        r'\$\{([^}]+)\}',
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value
    )


def _walk_expand(obj):
    """Recursively expand env-var placeholders in all string values of a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from a YAML file.

    Secrets (api_key, base_url, …) are kept in a .env file and referenced
    in config.yaml as  ${VAR_NAME}  placeholders.  This function:
      1. Loads .env (if it exists) so all  ${…}  references resolve.
      2. Parses config.yaml.
      3. Walks the parsed dict and expands every  ${VAR_NAME}  string.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to config.yaml next to this script.
    Returns:
        Nested dict of configuration values with env vars substituted.
    """
    # 1. Load .env (does NOT override vars already set in the shell environment)
    env_path = _DEFAULT_ENV_PATH
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info(f"Loaded .env from: {env_path}")
    else:
        logger.warning(
            f".env file not found at {env_path}. "
            "API key / base URL must be set as real environment variables."
        )

    # 2. Parse YAML
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Please create config.yaml (copy the template in this directory)."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 3. Expand ${VAR_NAME} placeholders
    cfg = _walk_expand(cfg)

    logger.info(f"Loaded config from: {path}")
    return cfg


@log_call(logger)
def read_precised_prompts(csv_path: str, num_rows: int = 100, seed: int = None) -> List[Dict]:
    all_rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                try:
                    precised_prompts_list = json.loads(row['precised_prompts'])
                    all_rows.append({
                        'csv_idx': idx,
                        'row_id': row.get('__row_id__', idx),
                        'original_prompt': row['prompt'],
                        'generic_prompt': row.get('generic_prompt', ''),
                        'precised_prompts': precised_prompts_list
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse precised_prompts for row {idx}: {e}")
                    continue

        logger.info(f"Read {len(all_rows)} total rows from {csv_path}")

        if seed is not None:
            random.seed(seed)
            if num_rows > len(all_rows):
                selected_rows = all_rows
            else:
                selected_rows = random.sample(all_rows, num_rows)
                selected_rows.sort(key=lambda x: int(x['row_id']))
        else:
            selected_rows = all_rows[:num_rows]

        prompts_data = []
        for idx, row in enumerate(selected_rows):
            row['row_idx'] = idx
            prompts_data.append(row)

        return prompts_data

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise


@log_call(logger)
def read_rows_from_csv(
    csv_path: str,
    prompt_col: str,
    is_json_list: bool = False,
    num_rows: Optional[int] = None,
    seed: Optional[int] = None,
    images_per_prompt: int = 10
) -> List[Dict]:
    """
    Reads prompts from a CSV column.
    
    Args:
        csv_path: Path to the CSV file.
        prompt_col: Name of the column containing the prompt(s).
        is_json_list: If True, parses the column value as a JSON list of strings.
                      If False, treats the column value as a single string and duplicates it.
        num_rows: Optional limit on number of rows to read.
        seed: Random seed for row sampling.
        images_per_prompt: Number of images to generate per prompt (only used if is_json_list=False).
                           If is_json_list=True, one image is generated per variant in the list.
    """
    all_rows: List[Dict] = []
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if prompt_col not in (reader.fieldnames or []):
                raise ValueError(f"CSV {csv_path} does not have column '{prompt_col}'. Columns: {reader.fieldnames}")

            for idx, row in enumerate(reader):
                raw_val = row[prompt_col]
                
                # Determine the list of prompts to run for this row
                if is_json_list:
                    try:
                        # Expecting a JSON list of strings, e.g. ["prompt1", "prompt2", ...]
                        # This typically comes from the 'enhanced/modified' prompts column
                        # where GPT-4 already generated N variations.
                        prompt_variants = json.loads(raw_val)
                        if not isinstance(prompt_variants, list):
                            logger.warning(f"Row {idx}: Parsed JSON is not a list. treating as single string.")
                            prompt_variants = [str(prompt_variants)]
                    except json.JSONDecodeError:
                        logger.warning(f"Row {idx}: Failed to parse JSON in '{prompt_col}'. Treating as raw string.")
                        prompt_variants = [raw_val]
                else:
                    # Single prompt string -> duplicate it N times
                    prompt_variants = [raw_val] * images_per_prompt

                # Create variant dicts with unique seeds
                # We generate a base seed for the row, then offset for each variant.
                base_seed = random.randint(0, 2**31 - 1)
                
                precised_prompts_dicts = []
                for i, p_text in enumerate(prompt_variants):
                    precised_prompts_dicts.append({
                        'prompt': p_text,
                        'seed': base_seed + i
                    })

                all_rows.append({
                    'csv_idx': idx,
                    'row_id': row.get('__row_id__', idx),
                    'original_prompt': row.get('prompt', raw_val), # fallback if 'prompt' col doesn't exist
                    'target_prompt_col': prompt_col,
                    'precised_prompts': precised_prompts_dicts,
                })

        logger.info(f"Read {len(all_rows)} total rows from {csv_path}")

        if num_rows is None:
            selected_rows = all_rows
        else:
            if seed is not None:
                random.seed(seed)
                if num_rows > len(all_rows):
                    selected_rows = all_rows
                else:
                    selected_rows = random.sample(all_rows, num_rows)
                    selected_rows.sort(key=lambda x: int(x['row_id']))
            else:
                selected_rows = all_rows[:num_rows]

        prompts_data: List[Dict] = []
        for idx, row in enumerate(selected_rows):
            row['row_idx'] = idx
            prompts_data.append(row)

        row_ids = [row['row_id'] for row in prompts_data]
        logger.info(f"Selected row IDs: {row_ids[:10]}{'...' if len(row_ids) > 10 else ''}")
        return prompts_data

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise


@log_call(logger)
def generate_image(cloud_cfg: Dict, prompt: str, seed: int, output_path: str) -> Dict:
    """
    Submit a text-to-image generation task to the cloud API, poll until it
    completes, download the resulting image, and save it to output_path.

    Workflow (mirrors test.py):
      1. POST  <base_url>/api/v2/generate  → receive task_id
      2. GET   <base_url>/api/v2/generate/<task_id>  (repeat until COMPLETED/ERROR)
      3. Download the image from generations[0]['url']
      4. Save raw bytes to output_path

    Args:
        cloud_cfg   : Sub-dict from config['cloud_api'] with keys:
                        api_key, base_url, model (dict), operation,
                        poll_interval, poll_timeout, download_timeout
        prompt      : Text prompt to send to the model.
        seed        : Integer seed (stored in metadata; some API backends honour it).
        output_path : Local filesystem path where the image will be saved.
    """
    api_key          = cloud_cfg['api_key']
    base_url         = cloud_cfg['base_url'].rstrip('/')
    model_cfg        = cloud_cfg['model']          # dict with 'name' and 'model' keys
    operation        = cloud_cfg.get('operation', 'Imagine')
    poll_interval    = int(cloud_cfg.get('poll_interval', 5))
    poll_timeout     = int(cloud_cfg.get('poll_timeout', 300))
    download_timeout = int(cloud_cfg.get('download_timeout', 60))

    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model":     model_cfg,
        "operation": operation,
        "args":      {"prompt": prompt},
    }

    try:
        logger.info(
            f"Submitting cloud generation  "
            f"model={model_cfg.get('model', '?')}  "
            f"seed={seed}  "
            f"prompt={prompt[:80]!r}"
        )

        # ── Step 1: Create generation task ──────────────────────────────────
        resp = requests.post(
            f"{base_url}/api/v2/generate",
            headers=auth_headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"API rejected generation request  "
                f"status={resp.status_code}  body={resp.text[:300]}"
            )

        task_id = resp.json()['id']
        logger.info(f"Task created: {task_id}")

        # ── Step 2: Poll for completion ──────────────────────────────────────
        elapsed = 0
        image_url = None
        while elapsed < poll_timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            status_resp = requests.get(
                f"{base_url}/api/v2/generate/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            data   = status_resp.json()
            status = data.get('status')
            logger.debug(f"[{task_id}] status={status}  elapsed={elapsed}s")

            if status == 'COMPLETED':
                image_url = data['generations'][0]['url']
                cost = data.get('usage', {}).get('cost', 'N/A')
                logger.info(
                    f"✓ Task completed  id={task_id}  cost={cost}c  "
                    f"url={image_url}"
                )
                break
            elif status in ('ERROR', 'CANCELLED'):
                raise RuntimeError(
                    f"Task {task_id} ended with status '{status}': "
                    f"{data.get('error', 'no detail')}"
                )

        if image_url is None:
            raise TimeoutError(
                f"Task {task_id} did not complete within {poll_timeout}s."
            )

        # ── Step 3: Download generated image ────────────────────────────────
        logger.info(f"Downloading image from: {image_url}")
        img_resp = requests.get(image_url, timeout=download_timeout)
        img_resp.raise_for_status()

        # ── Step 4: Save to disk ─────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(img_resp.content)

        logger.info(
            f"✓ Image saved: {output_path}  ({len(img_resp.content) // 1024} KB)"
        )

        return {
            'status':          'success',
            'output_path':     output_path,
            'original_prompt': prompt,
            'seed':            seed,
            'task_id':         task_id,
            'image_url':       image_url,
            'api_model':       model_cfg.get('model', ''),
        }

    except Exception as e:
        logger.error(f"✗ Failed to generate image: {e}")
        return {
            'status':          'error',
            'error':           str(e),
            'output_path':     output_path,
            'original_prompt': prompt,
            'seed':            seed,
        }


@log_call(logger, level=logging.DEBUG)
def generate_images_for_row(cloud_cfg: Dict, row_data: Dict, output_dir: Path) -> List[Dict]:
    """
    Generate all image variants for one CSV row by calling the cloud API.

    Each element of row_data['precised_prompts'] is a dict:
        {'prompt': str, 'seed': int}
    The unique seed per variant is stored in the result metadata; some API
    backends use it to produce diverse outputs for the same prompt.
    """
    results = []
    row_idx = row_data['row_idx']
    row_id  = row_data['row_id']
    precised_prompts = row_data['precised_prompts']

    row_dir = output_dir / f"row_{row_id}"
    row_dir.mkdir(parents=True, exist_ok=True)

    log_subsection(
        logger,
        f"Row {row_idx}  (CSV ID: {row_id})  ·  {len(precised_prompts)} variants"
    )

    for variant_idx, variant in enumerate(precised_prompts):
        prompt = variant['prompt']
        seed   = variant['seed']

        output_filename = f"row_{row_id}_variant_{variant_idx:02d}.png"
        output_path     = row_dir / output_filename

        result = generate_image(cloud_cfg, prompt, seed, str(output_path))
        result['row_idx']     = row_idx
        result['row_id']      = row_id
        result['variant_idx'] = variant_idx
        results.append(result)

    return results


@log_call(logger)
def process_parallel(cloud_cfg: Dict, prompts_data: List[Dict], output_dir: Path, num_workers: int = 4) -> List[Dict]:
    """
    Process multiple CSV rows concurrently using a thread pool.

    Cloud API calls are I/O-bound (submit → poll → download), so parallelism
    gives a significant throughput improvement even with a modest worker count.
    """
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_row = {
            executor.submit(generate_images_for_row, cloud_cfg, row_data, output_dir): row_data
            for row_data in prompts_data
        }
        for future in as_completed(future_to_row):
            row_data = future_to_row[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(
                    f"✓ Completed row {row_data['row_idx']} "
                    f"(CSV ID: {row_data['row_id']}): {len(results)} images"
                )
            except Exception as e:
                logger.error(
                    f"✗ Row {row_data['row_idx']} "
                    f"(CSV ID: {row_data['row_id']}) failed: {e}"
                )
    return all_results


def save_results_metadata(metadata: Dict, output_path: str):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Results metadata saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results metadata: {e}")


def print_summary(results: List[Dict]):
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful
    logger.info(f"\n{'='*80}")
    logger.info("GENERATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total images attempted: {total}")
    if total == 0:
        logger.warning("No images were attempted.")
        return
    logger.info(f"Successful: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")


# ----------------------------
# Metrics code
# ----------------------------

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def detect_face(image_paths, SAVE_DETECTED_AT, default_max_size=800, size=300, padding=0.25):
    face_detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(os.path.join(DLIB_MODEL_DIR, "shape_predictor_5_face_landmarks.dat"))

    for index, image_path in enumerate(image_paths):
        if index % 100 == 0:
            print('Processing image: %d/%d' % (index, len(image_paths)))
        try:
            img = dlib.load_rgb_image(image_path)
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        old_height, old_width, _ = img.shape
        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height = int(default_max_size * old_width / old_height), default_max_size

        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = face_detector(img, 1)
        if len(dets) == 0:
            print("No face detected in: {}".format(image_path))
            continue

        faces = dlib.full_object_detections()
        for rect in dets:
            faces.append(sp(img, rect))

        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, image in enumerate(images):
            img_name = os.path.basename(image_path)
            base, ext = os.path.splitext(img_name)
            face_name = os.path.join(SAVE_DETECTED_AT, f"{base}_face{idx}{ext}")
            dlib.save_image(image, face_name)


def is_image_file(path):
    if not os.path.isfile(path):
        return False
    _, ext = os.path.splitext(path.lower())
    return ext in {'.jpg', '.jpeg', '.png', '.bmp'}


def predict_age_gender_race(save_prediction_at, imgs_path='detected_faces/'):
    img_names = []
    if os.path.exists(imgs_path):
        for entry in os.listdir(imgs_path):
            full = os.path.join(imgs_path, entry)
            if is_image_file(full):
                img_names.append(full)
    else:
        print(f"Directory {imgs_path} does not exist.")
        return

    device = torch.device("cpu")  # Forced CPU: keep both A100s free for FLUX / Qwen servers

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(
    torch.load(os.path.join(FAIRFACE_MODEL_DIR, "res34_fair_align_multi_7_20190809.pt"), map_location=device))

    model_fair_7 = model_fair_7.to(device).eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face_names = []
    race_preds_fair, gender_preds_fair, age_preds_fair = [], [], []
    race_scores_fair, gender_scores_fair, age_scores_fair = [], [], []

    for i, img_name in enumerate(img_names):
        if i % 100 == 0:
            print('Predicting attributes: %d/%d' % (i, len(img_names)))

        try:
            image_raw = dlib.load_rgb_image(img_name)
        except Exception as e:
            print(f"Warning: skipping {img_name} (load error: {e})")
            continue

        face_names.append(img_name)
        image = trans(image_raw).view(1, 3, 224, 224).to(device)

        outputs = model_fair_7(image).cpu().detach().numpy().squeeze()
        race_score = softmax(outputs[:7])
        gender_score = softmax(outputs[7:9])
        age_score = softmax(outputs[9:18])

        race_preds_fair.append(np.argmax(race_score))
        gender_preds_fair.append(np.argmax(gender_score))
        age_preds_fair.append(np.argmax(age_score))
        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

    result = pd.DataFrame({
        'face_name_align': face_names,
        'race_preds_fair': race_preds_fair,
        'gender_preds_fair': gender_preds_fair,
        'age_preds_fair': age_preds_fair,
        'race_scores_fair': race_scores_fair,
        'gender_scores_fair': gender_scores_fair,
        'age_scores_fair': age_scores_fair
    })

    result['race'] = result['race_preds_fair'].apply(lambda x: race_labels[x])
    result['gender'] = result['gender_preds_fair'].apply(lambda x: gender_labels[x])
    result['age'] = result['age_preds_fair'].apply(lambda x: age_labels[x])

    result = result.drop(columns=[
        'race_preds_fair', 'gender_preds_fair', 'age_preds_fair'
    ])

    result.to_csv(save_prediction_at, index=False)
    print("Results saved to:", save_prediction_at)


race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
gender_labels = ['Male', 'Female']
age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

attributes_dict = {
    'race': race_labels,
    'gender': gender_labels,
    'age': age_labels
}

combinations_to_analyze = [
    ['race'], ['gender'], ['age'],
    ['race', 'gender'], ['race', 'age'], ['gender', 'age'],
    ['race', 'gender', 'age']
]


def calculate_combined_bias_metrics(prediction_csv_path, bias_w_csv_path='bias_w_metrics.csv', bias_p_csv_path='bias_p_metrics.csv'):
    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {prediction_csv_path} was not found.")
        return

    df['original_image'] = df['face_name_align'].apply(lambda x: os.path.basename(x).split('_face')[0])
    bias_w_records = []
    bias_p_records = []

    for attribute_group in combinations_to_analyze:
        attr_group_name = "+".join(attribute_group)

        if len(attribute_group) == 1:
            attr_type = attribute_group[0]
            freq_w = df[attr_type].value_counts(normalize=True)
        else:
            freq_w = df.groupby(attribute_group).size() / len(df)

        if len(attribute_group) == 1:
            na = len(attributes_dict[attribute_group[0]])
        elif len(attribute_group) == 2:
            na = len(attributes_dict[attribute_group[0]]) * len(attributes_dict[attribute_group[1]])
        else:
            na = len(attributes_dict[attribute_group[0]]) * len(attributes_dict[attribute_group[1]]) * len(attributes_dict[attribute_group[2]])

        bias_w_sum = 0
        if len(attribute_group) == 1:
            all_combs = attributes_dict[attribute_group[0]]
        elif len(attribute_group) == 2:
            all_combs = [(a, b) for a in attributes_dict[attribute_group[0]] for b in attributes_dict[attribute_group[1]]]
        else:
            all_combs = [(a, b, c) for a in attributes_dict[attribute_group[0]] for b in attributes_dict[attribute_group[1]] for c in attributes_dict[attribute_group[2]]]

        for comb in all_combs:
            if isinstance(comb, str):
                freq_a_w = freq_w.get(comb, 0)
            else:
                try:
                    freq_a_w = freq_w.loc[comb]
                except KeyError:
                    freq_a_w = 0
            bias_w_sum += (freq_a_w - (1 / na)) ** 2

        bias_w = np.sqrt((1 / na) * bias_w_sum)
        bias_w_records.append({
            'Attribute_Group': attr_group_name,
            'Bias-W': bias_w
        })

        for image_name, group in df.groupby('original_image'):
            if len(group) > 1:
                if len(attribute_group) == 1:
                    attr_type = attribute_group[0]
                    freq_p = group[attr_type].value_counts(normalize=True)
                else:
                    freq_p = group.groupby(attribute_group).size() / len(group)

                bias_p_sum = 0
                for comb in all_combs:
                    if isinstance(comb, str):
                        freq_a_p = freq_p.get(comb, 0)
                    else:
                        try:
                            freq_a_p = freq_p.loc[comb]
                        except KeyError:
                            freq_a_p = 0
                    bias_p_sum += (freq_a_p - (1 / na)) ** 2

                bias_p = np.sqrt((1 / na) * bias_p_sum)

                bias_p_records.append({
                    'Original_Image': image_name,
                    'Attribute_Group': attr_group_name,
                    'Bias-P': bias_p
                })

    bias_w_df = pd.DataFrame(bias_w_records)
    bias_w_df.to_csv(bias_w_csv_path, index=False)
    print(f"Bias-W metrics saved to: {bias_w_csv_path}")

    bias_p_df = pd.DataFrame(bias_p_records)
    bias_p_df.to_csv(bias_p_csv_path, index=False)
    print(f"Bias-P metrics saved to: {bias_p_csv_path}")


def calculate_ens_metrics(prediction_csv_path, output_csv='ens_metrics.csv'):
    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {prediction_csv_path} was not found.")
        return

    ens_records = []
    for attribute_group in combinations_to_analyze:
        attrs = attribute_group
        if len(attrs) == 1:
            p = df[attrs[0]].value_counts(normalize=True)
        else:
            p = df.groupby(attrs).size() / len(df)

        p_list = p.values
        sum_p_ln_p = 0.0
        for p_g in p_list:
            if p_g > 0:
                sum_p_ln_p += p_g * np.log(p_g)

        ens_value = float(np.exp(-sum_p_ln_p))
        ens_records.append({
            'Attribute': "+".join(attrs),
            'ENS': ens_value
        })

    ens_df = pd.DataFrame(ens_records)
    ens_df.to_csv(output_csv, index=False)
    print(f"ENS metrics saved to: {output_csv}")


def calculate_kl_divergence(prediction_csv_path, output_csv='kl_divergence_metrics.csv', reference_distribution=None):
    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {prediction_csv_path} was not found.")
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
                labels = list(itertools.product(*(attributes_dict[attr] for attr in attribute_group)))
            ref_dist = pd.Series({label: 1/len(labels) for label in labels})

        all_labels = sorted(set(gen_dist.index) | set(ref_dist.index))
        gen_probs = np.array([gen_dist.get(label, 0) for label in all_labels], dtype=float)
        ref_probs = np.array([ref_dist.get(label, 0) for label in all_labels], dtype=float)

        eps = 1e-12
        gen_probs = np.clip(gen_probs, eps, 1)
        ref_probs = np.clip(ref_probs, eps, 1)

        kl_value = np.sum(gen_probs * np.log(gen_probs / ref_probs))
        kl_records.append({
            'Attribute_Group': attr_group_name,
            'KL_Divergence': kl_value
        })

    kl_df = pd.DataFrame(kl_records)
    kl_df.to_csv(output_csv, index=False)
    print(f"KL Divergence metrics saved to: {output_csv}")


# --- NEW FUNCTION: CALCULATE ICAD METRICS (ADAPTED FROM SECOND SCRIPT) ---
def calculate_icad_metrics(prediction_csv_path, output_csv='icad_metrics.csv'):
    """
    Calculates Stratified ICAD (Intra-Class Average Distance) using CLIP embeddings.
    """
    try:
        # Check if CLIP is available
        import clip
    except ImportError:
        logger.error("CLIP module not found. Skipping ICAD calculation.")
        return

    try:
        df = pd.read_csv(prediction_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {prediction_csv_path} was not found.")
        return

    if df.empty:
        print("DataFrame is empty, skipping ICAD.")
        return

    device = "cpu"  # Forced CPU: keep both A100s free for FLUX / Qwen servers
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Dictionary to store embeddings to avoid re-computation
    path_to_emb = {}
    
    # We use 'face_name_align' as the column for image paths (from First Script logic)
    image_paths = df['face_name_align'].tolist()
    
    print(f"Generating CLIP embeddings for {len(image_paths)} faces...")
    with torch.no_grad():
        for path in image_paths:
            if path in path_to_emb:
                continue
            try:
                img = Image.open(path).convert("RGB")
                img_input = preprocess(img).unsqueeze(0).to(device)
                emb = model.encode_image(img_input)
                path_to_emb[path] = emb.cpu().numpy().reshape(-1)
            except Exception as e:
                # logger.warning(f"Could not process image for CLIP: {path} - {e}")
                continue

    final_metrics = []

    # Iterate through the same combinations used in other metrics
    for attribute_group in combinations_to_analyze:
        comb_name = "+".join(attribute_group)
        
        # Create a temporary group column for stratification
        if len(attribute_group) == 1:
            df['temp_group'] = df[attribute_group[0]]
        else:
            df['temp_group'] = df[attribute_group].apply(lambda x: '_'.join(x.astype(str)), axis=1)

        subgroup_icads = []
        
        # Stratified calculation: Calculate ICAD for each subgroup (e.g., White Males, Black Females)
        for g_name, g_df in df.groupby('temp_group'):
            embs = [path_to_emb[p] for p in g_df['face_name_align'] if p in path_to_emb]
            
            # We need at least 2 points to calculate intra-class distance
            if len(embs) > 1:
                X = np.array(embs)
                # Calculate distance of each point from the centroid
                dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
                subgroup_icads.append(dists.mean())
        
        # Average the subgroup ICADs to get the final metric for this attribute combination
        avg_icad = np.mean(subgroup_icads) if subgroup_icads else 0.0
        
        final_metrics.append({
            'Attribute': comb_name, 
            'Stratified_ICAD': avg_icad
        })

    # cleanup temp column
    if 'temp_group' in df.columns:
        df.drop(columns=['temp_group'], inplace=True)

    icad_df = pd.DataFrame(final_metrics)
    icad_df.to_csv(output_csv, index=False)
    print(f"ICAD metrics saved to: {output_csv}")
# -----------------------------------------------------------------------


def process_paths_and_predict(img_paths, tag):
    """
    Orchestrator for metrics calculation.
    """
    log_section(logger, f"Metrics pipeline  ·  tag={tag}  ·  {len(img_paths)} images")

    detected_dir = os.path.join(BASE_DIR, f"detected_faces_{tag}")
    ensure_dir(detected_dir)

    output_csv = os.path.join(BASE_DIR, f"test_outputs_{tag}.csv")
    bias_p_csv = os.path.join(BASE_DIR, f"bias_p_metrics_{tag}.csv")
    bias_w_csv = os.path.join(BASE_DIR, f"bias_w_metrics_{tag}.csv")
    ens_csv = os.path.join(BASE_DIR, f"ens_{tag}.csv")
    kl_csv = os.path.join(BASE_DIR, f"kl_divergence_metrics_{tag}.csv")
    icad_csv = os.path.join(BASE_DIR, f"icad_metrics_{tag}.csv")

    img_list_csv = os.path.join(BASE_DIR, f"image_paths_{tag}.csv")
    pd.DataFrame({'img_path': img_paths}).to_csv(img_list_csv, index=False)

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


# ----------------------------
# Orchestration
# ----------------------------

def _check_metrics_models(metrics_base_dir: Path):
    fair = metrics_base_dir / "res34_fair_align_multi_7_20190809.pt"
    dlib_model = metrics_base_dir / "shape_predictor_5_face_landmarks.dat"
    missing = []
    if not fair.exists():
        missing.append(str(fair))
    if not dlib_model.exists():
        missing.append(str(dlib_model))
    if missing:
        raise FileNotFoundError(
            "Missing required metrics model files:\n - " + "\n - ".join(missing) +
            "\n\nExpected structure:\n"
            f"  {metrics_base_dir}/res34_fair_align_multi_7_20190809.pt\n"
            f"  {metrics_base_dir}/shape_predictor_5_face_landmarks.dat\n"
        )


@log_call(logger)
def generate_images_from_csv(
    cloud_cfg: Dict,
    csv_path: str,
    tag: str,
    output_root: Path,
    prompt_col: str,
    num_rows: Optional[int],
    seed: Optional[int],
    images_per_prompt: int,
    workers: int,
    sequential: bool,
    is_json_list: bool = False
) -> List[Dict]:
    """
    Read prompts from a CSV, generate images via the cloud API for each,
    save metadata, and return all result dicts.
    """
    log_subsection(logger, f"[{tag}] loading prompts  ·  csv={csv_path}  col={prompt_col}")
    prompts_data = read_rows_from_csv(
        csv_path=csv_path,
        prompt_col=prompt_col,
        is_json_list=is_json_list,
        num_rows=num_rows,
        seed=seed,
        images_per_prompt=images_per_prompt
    )

    if not prompts_data:
        raise ValueError(f"No prompts loaded from {csv_path}")

    out_dir = output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "sequential" if sequential else f"parallel  workers={workers}"
    log_subsection(logger, f"[{tag}] generating images  ·  {len(prompts_data)} rows  ·  {mode}")

    start_time = time.time()
    with log_timer(logger, f"[{tag}] full generation pass  ({len(prompts_data)} rows)"):
        if sequential:
            all_results = []
            for row_data in prompts_data:
                results = generate_images_for_row(cloud_cfg, row_data, out_dir)
                all_results.extend(results)
        else:
            all_results = process_parallel(cloud_cfg, prompts_data, out_dir, num_workers=workers)

    elapsed = time.time() - start_time

    metadata = {
        'configuration': {
            'api_base_url':      cloud_cfg.get('base_url', ''),
            'api_model':         cloud_cfg.get('model', {}).get('model', ''),
            'csv_file':          csv_path,
            'prompt_col':        prompt_col,
            'num_rows':          num_rows,
            'seed':              seed,
            'output_dir':        str(out_dir),
            'workers':           workers if not sequential else 1,
            'sequential':        sequential,
            'images_per_prompt': images_per_prompt,
            'tag':               tag,
        },
        'results': all_results,
        'summary': {
            'total_images': len(all_results),
            'successful': sum(1 for r in all_results if r['status'] == 'success'),
            'failed': sum(1 for r in all_results if r['status'] == 'error'),
            'total_time_seconds': elapsed,
            'average_time_per_image': elapsed / len(all_results) if all_results else 0
        }
    }
    results_file = out_dir / "generation_results.json"
    save_results_metadata(metadata, str(results_file))
    print_summary(all_results)

    return all_results


def successful_image_paths(results: List[Dict]) -> List[str]:
    return [r['output_path'] for r in results if r.get('status') == 'success' and r.get('output_path')]


def collect_images_from_tag_dir(base_dir: Path, tag: str) -> List[str]:
    """
    Recursively collect all image files for a given tag from an already-generated
    output folder.  Search order:

      1. <base_dir>/generated_images/<tag>/   ← standard layout produced by this script
      2. <base_dir>/<tag>/                     ← flat layout (no generated_images/ wrapper)
      3. <base_dir>/                           ← entire folder (only when tag not found above)

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


def model_tag_from_config(cloud_cfg: Dict) -> str:
    """Derive a short snake_case tag from the cloud config model name."""
    model_name = cloud_cfg.get('model', {}).get('model', 'model')
    # Replace non-alphanumeric chars with underscores and lowercase
    import re
    tag = re.sub(r'[^a-z0-9]', '_', model_name.lower()).strip('_')
    return tag or "model"


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate images via the Metis cloud API for ORIGINAL and ENHANCED '
            'prompt CSVs, then compute bias/diversity metrics.'
        )
    )

    # -----------------------------------------------------------------------
    # Config file
    # -----------------------------------------------------------------------
    parser.add_argument(
        '--config', type=str, default=None,
        help=(
            'Path to the YAML config file.  '
            'Defaults to config.yaml in the same directory as this script.  '
            'All settings (API key, model, CSV paths, workers, …) live there; '
            'CLI args below override individual config values when provided.'
        )
    )

    # -----------------------------------------------------------------------
    # Cloud API overrides (all optional — defaults come from config.yaml)
    # -----------------------------------------------------------------------
    parser.add_argument('--api-key',   type=str, default=None, help='Override cloud_api.api_key from config')
    parser.add_argument('--base-url',  type=str, default=None, help='Override cloud_api.base_url from config')
    parser.add_argument('--api-model', type=str, default=None, help='Override cloud_api.model.model from config (e.g. nano-banana)')

    # -----------------------------------------------------------------------
    # Evaluate-only mode
    # -----------------------------------------------------------------------
    parser.add_argument(
        '--evaluate-only', action='store_true',
        help=(
            'Skip image generation and run metrics on images that already exist '
            'in --eval-folder.  Reads evaluate_only settings from config if not '
            'overridden here.'
        )
    )
    parser.add_argument('--eval-folder', type=str, default=None,
                        help='Root folder with already-generated images (overrides config)')
    parser.add_argument('--eval-tags',   type=str, default=None,
                        help="Comma-separated tags to evaluate (overrides config, default: 'original,enhanced')")

    # -----------------------------------------------------------------------
    # CSV / prompt overrides
    # -----------------------------------------------------------------------
    parser.add_argument('--csv-file',            type=str,   default=None, help='Single CSV for both passes (overrides config)')
    parser.add_argument('--original-csv',        type=str,   default=None, help='CSV with original prompts (overrides config)')
    parser.add_argument('--enhanced-csv',        type=str,   default=None, help='CSV with enhanced prompts (overrides config)')
    parser.add_argument('--original-prompt-col', type=str,   default=None, help='Column name for original prompts (overrides config)')
    parser.add_argument('--enhanced-prompt-col', type=str,   default=None, help='Column name for enhanced prompts (overrides config)')
    parser.add_argument('--enhanced-is-json',    action=argparse.BooleanOptionalAction, default=None,
                        help='Parse enhanced prompt column as JSON list (overrides config)')
    parser.add_argument('--num-rows',  type=int,  default=None, help='Limit rows processed (overrides config)')
    parser.add_argument('--seed',      type=int,  default=None, help='Random seed for row sampling (overrides config)')

    # -----------------------------------------------------------------------
    # Generation overrides
    # -----------------------------------------------------------------------
    parser.add_argument('--images-per-prompt', type=int,  default=None, help='Images per prompt (overrides config)')
    parser.add_argument('--output-dir',        type=str,  default=None, help='Output root folder (overrides config)')
    parser.add_argument('--workers',           type=int,  default=None, help='Parallel worker threads (overrides config)')
    parser.add_argument('--sequential',        action='store_true',     help='Force sequential processing (overrides config)')

    # -----------------------------------------------------------------------
    # Metrics override
    # -----------------------------------------------------------------------
    parser.add_argument('--metrics-base-dir', type=str, default=None,
                        help='Dir containing FairFace + dlib model files (overrides config)')

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── Apply CLI overrides onto the loaded config ───────────────────────────
    cloud_cfg = cfg['cloud_api']
    if args.api_key:
        cloud_cfg['api_key'] = args.api_key
    if args.base_url:
        cloud_cfg['base_url'] = args.base_url
    if args.api_model:
        cloud_cfg.setdefault('model', {})['model'] = args.api_model

    gen_cfg = cfg.get('generation', {})
    csv_cfg = cfg.get('csv', {})
    metrics_cfg = cfg.get('metrics', {})
    eo_cfg = cfg.get('evaluate_only', {})

    images_per_prompt = args.images_per_prompt or gen_cfg.get('images_per_prompt', 10)
    output_dir_str    = args.output_dir        or gen_cfg.get('output_dir', 'generated_images')
    workers           = args.workers           or gen_cfg.get('workers', 4)
    sequential        = args.sequential        or gen_cfg.get('sequential', False)

    num_rows          = args.num_rows          if args.num_rows  is not None else csv_cfg.get('num_rows')
    seed              = args.seed              if args.seed      is not None else csv_cfg.get('seed')
    enhanced_is_json  = args.enhanced_is_json  if args.enhanced_is_json is not None else csv_cfg.get('enhanced_is_json', True)

    metrics_base_dir = args.metrics_base_dir or metrics_cfg.get('base_dir', 'models')

    global BASE_DIR, FAIRFACE_MODEL_DIR, DLIB_MODEL_DIR
    FAIRFACE_MODEL_DIR = os.path.abspath(metrics_base_dir)
    DLIB_MODEL_DIR     = os.path.abspath(metrics_base_dir)

    _check_metrics_models(Path(FAIRFACE_MODEL_DIR))

    # ── Build a concise run tag from the model name + date ───────────────────
    model_tag      = model_tag_from_config(cloud_cfg)
    date_tag       = datetime.now().strftime('%Y%m%d')
    run_output_tag = f"{model_tag}_output_{date_tag}"

    # ── All metrics outputs go under eval_outputs/<run_tag>/ ────────────────
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_outputs', run_output_tag)
    os.makedirs(BASE_DIR, exist_ok=True)

    logger.info(f"Cloud API  base_url={cloud_cfg['base_url']}  model={cloud_cfg.get('model', {})}")
    logger.info(f"Run output tag: {run_output_tag}")

    # -----------------------------------------------------------------------
    # Evaluate-only branch
    # -----------------------------------------------------------------------
    evaluate_only = args.evaluate_only or eo_cfg.get('enabled', False)
    if evaluate_only:
        eval_folder_str = args.eval_folder or eo_cfg.get('eval_folder')
        eval_tags_str   = args.eval_tags   or eo_cfg.get('eval_tags', 'original,enhanced')

        if not eval_folder_str:
            parser.error(
                "--eval-folder (or evaluate_only.eval_folder in config.yaml) "
                "is required when evaluate-only mode is active."
            )

        eval_folder = Path(eval_folder_str).resolve()
        if not eval_folder.is_dir():
            parser.error(f"eval-folder does not exist or is not a directory: {eval_folder}")

        BASE_DIR = str(eval_folder / "eval_outputs")
        os.makedirs(BASE_DIR, exist_ok=True)

        tags = [t.strip() for t in eval_tags_str.split(',') if t.strip()]
        logger.info(f"evaluate-only mode  ·  folder={eval_folder}  ·  tags={tags}")

        evaluated_any = False
        for tag in tags:
            img_paths = collect_images_from_tag_dir(eval_folder, tag)
            if not img_paths:
                logger.warning(f"No images found for tag '{tag}' — skipping.")
                continue
            log_section(logger, f"Evaluate-only  ·  tag={tag}  ·  {len(img_paths)} images")
            process_paths_and_predict(img_paths, tag)
            evaluated_any = True

        if not evaluated_any:
            logger.error(
                f"No images found under {eval_folder} for tags {tags}. "
                "Check the folder structure:\n"
                "  <eval-folder>/generated_images/<tag>/row_*/  OR  <eval-folder>/<tag>/"
            )

        logger.info("\n\u2705 Evaluate-only run complete.")
        logger.info(f"Metrics written under: {Path(BASE_DIR).resolve()}")
        return

    # -----------------------------------------------------------------------
    # Normal generation + evaluation path
    # -----------------------------------------------------------------------

    # Determine CSV paths (CLI > config)
    single_csv    = args.csv_file     or csv_cfg.get('file')
    original_csv  = args.original_csv or csv_cfg.get('original_csv')
    enhanced_csv  = args.enhanced_csv or csv_cfg.get('enhanced_csv')
    orig_col      = args.original_prompt_col or csv_cfg.get('original_prompt_col', 'generic_prompt')
    enh_col       = args.enhanced_prompt_col or csv_cfg.get('enhanced_prompt_col', 'modified_prompts')

    if single_csv:
        original_csv = single_csv
        enhanced_csv = single_csv
        logger.info(f"Using single CSV for both passes: {single_csv}")
    else:
        if not original_csv or not enhanced_csv:
            parser.error(
                "Either csv.file OR both csv.original_csv and csv.enhanced_csv "
                "must be set in config.yaml (or passed via --csv-file / --original-csv / --enhanced-csv)."
            )

    # Generated images go under <output_dir>/<run_tag>/
    output_root = Path(output_dir_str) / run_output_tag
    output_root.mkdir(parents=True, exist_ok=True)

    # 1) ORIGINAL prompts
    log_section(logger, "Original prompts  \u00b7  cloud image generation")
    original_results = generate_images_from_csv(
        cloud_cfg=cloud_cfg,
        csv_path=original_csv,
        tag="original",
        output_root=output_root,
        prompt_col=orig_col,
        is_json_list=False,
        num_rows=num_rows,
        seed=seed,
        images_per_prompt=images_per_prompt,
        workers=workers,
        sequential=sequential,
    )
    original_img_paths = successful_image_paths(original_results)
    if original_img_paths:
        logger.info(
            f"Running metrics for ORIGINAL set on "
            f"{len(original_img_paths)} successfully generated images..."
        )
        process_paths_and_predict(original_img_paths, "original")
    else:
        logger.warning("No successful original images generated. Skipping metrics.")

    # 2) ENHANCED prompts
    log_section(logger, "Enhanced prompts  \u00b7  cloud image generation")
    enhanced_results = generate_images_from_csv(
        cloud_cfg=cloud_cfg,
        csv_path=enhanced_csv,
        tag="enhanced",
        output_root=output_root,
        prompt_col=enh_col,
        is_json_list=enhanced_is_json,
        num_rows=num_rows,
        seed=seed,
        images_per_prompt=images_per_prompt,
        workers=workers,
        sequential=sequential,
    )
    enhanced_img_paths = successful_image_paths(enhanced_results)
    if enhanced_img_paths:
        logger.info(
            f"Running metrics for ENHANCED set on "
            f"{len(enhanced_img_paths)} successfully generated images..."
        )
        process_paths_and_predict(enhanced_img_paths, "enhanced")
    else:
        logger.warning("No successful enhanced images generated. Skipping metrics.")

    logger.info("\n\u2705 Done.")
    logger.info(f"Images written under:  {output_root.resolve()}")
    logger.info(f"Metrics written under: {Path(BASE_DIR).resolve()}")


if __name__ == '__main__':
    main()