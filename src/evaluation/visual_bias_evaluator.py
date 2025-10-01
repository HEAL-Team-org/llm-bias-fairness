"""Visual Bias Evaluation Module for LLM Bias & Fairness Pipeline.

This module implements visual bias and diversity evaluation metrics for generated images,
including face detection, demographic prediction, and bias measurements.
Based on FairFace and demographic analysis methods.
"""

import itertools
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import dlib
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DemographicPrediction:
    """Demographic prediction for a single face."""

    face_path: str
    original_image: str
    race: str
    gender: str
    age: str
    race_scores: np.ndarray
    gender_scores: np.ndarray
    age_scores: np.ndarray


@dataclass
class BiasMetrics:
    """Container for bias evaluation results."""

    bias_w_metrics: pd.DataFrame
    bias_p_metrics: pd.DataFrame
    ens_metrics: pd.DataFrame
    kl_divergence_metrics: pd.DataFrame


class VisualBiasEvaluator:
    """Visual bias and diversity evaluator for generated images."""

    # Demographic categories (matching notebook)
    RACE_LABELS = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
    GENDER_LABELS = ["Male", "Female"]
    AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    ATTRIBUTES_DICT = {
        "race": RACE_LABELS,
        "gender": GENDER_LABELS,
        "age": AGE_LABELS
    }

    # Attribute combinations to analyze
    COMBINATIONS = [
        ["race"], ["gender"], ["age"],
        ["race", "gender"], ["race", "age"], ["gender", "age"],
        ["race", "gender", "age"]
    ]

    def __init__(
        self,
        fairface_model_path: Optional[str] = None,
        dlib_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize the evaluator.
        
        Args:
            fairface_model_path: Path to FairFace model file
            dlib_model_path: Path to dlib shape predictor model
            device: Device for PyTorch (cuda/cpu)

        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch, dlib, and related dependencies not available. "
                "Install with: pip install torch torchvision dlib opencv-python"
            )

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fairface_model_path = fairface_model_path
        self.dlib_model_path = dlib_model_path
        self._models_loaded = False

        # Models will be loaded on first use
        self.fairface_model = None
        self.face_detector = None
        self.shape_predictor = None
        self.transform = None

    def _load_models(self):
        """Load FairFace and dlib models on demand."""
        if self._models_loaded:
            return

        logger.info("Loading FairFace and dlib models...")

        # Load FairFace model
        if self.fairface_model_path and os.path.exists(self.fairface_model_path):
            self.fairface_model = torchvision.models.resnet34(pretrained=True)
            self.fairface_model.fc = nn.Linear(self.fairface_model.fc.in_features, 18)
            self.fairface_model.load_state_dict(
                torch.load(self.fairface_model_path, map_location=self.device)
            )
            self.fairface_model = self.fairface_model.to(self.device).eval()
        else:
            logger.warning(
                "FairFace model not available. "
                "Download from: https://github.com/joojs/fairface"
            )
            # Use a mock model for testing
            self._create_mock_model()

        # Load dlib models
        self.face_detector = dlib.get_frontal_face_detector()

        if self.dlib_model_path and os.path.exists(self.dlib_model_path):
            self.shape_predictor = dlib.shape_predictor(self.dlib_model_path)
        else:
            logger.warning(
                "dlib shape predictor not available. "
                "Download from: http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
            )

        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._models_loaded = True
        logger.info("Models loaded successfully")

    def _create_mock_model(self):
        """Create a mock model for testing when FairFace is not available."""
        self.fairface_model = torchvision.models.resnet34()
        self.fairface_model.fc = nn.Linear(self.fairface_model.fc.in_features, 18)
        self.fairface_model = self.fairface_model.to(self.device).eval()
        logger.info("Using mock FairFace model for testing")

    def detect_faces(self, image_path: str, output_dir: str) -> List[str]:
        """Detect faces in an image and save aligned face crops.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save face crops
            
        Returns:
            List of paths to saved face crops

        """
        self._load_models()

        if not self.face_detector:
            logger.warning("Face detector not available, skipping face detection")
            return []

        try:
            # Load image
            img = dlib.load_rgb_image(image_path)

            # Detect faces
            dets = self.face_detector(img, 1)
            if len(dets) == 0:
                logger.info(f"No faces detected in: {image_path}")
                return []

            # Extract face regions
            faces = dlib.full_object_detections()
            if self.shape_predictor:
                for rect in dets:
                    faces.append(self.shape_predictor(img, rect))

                # Get aligned face chips
                face_crops = dlib.get_face_chips(img, faces, size=224, padding=0.25)
            else:
                # Fallback: simple bounding box crops
                face_crops = []
                for rect in dets:
                    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                    face_crop = img[y:y+h, x:x+w]
                    face_crops.append(face_crop)

            # Save face crops
            os.makedirs(output_dir, exist_ok=True)
            saved_paths = []

            img_name = os.path.basename(image_path)
            base, ext = os.path.splitext(img_name)

            for idx, face_crop in enumerate(face_crops):
                face_name = os.path.join(output_dir, f"{base}_face{idx}.png")
                if isinstance(face_crop, np.ndarray):
                    # Convert numpy array to PIL Image and save
                    face_img = Image.fromarray(face_crop)
                    face_img.save(face_name)
                else:
                    # dlib face chip - save directly
                    dlib.save_image(face_crop, face_name)
                saved_paths.append(face_name)

            return saved_paths

        except Exception as e:
            logger.error(f"Face detection failed for {image_path}: {e}")
            return []

    def predict_demographics(self, face_path: str) -> Optional[DemographicPrediction]:
        """Predict demographics for a single face image.
        
        Args:
            face_path: Path to face crop image
            
        Returns:
            DemographicPrediction or None if prediction fails

        """
        self._load_models()

        if not self.fairface_model:
            logger.warning("FairFace model not available, using random predictions")
            return self._mock_demographic_prediction(face_path)

        try:
            # Load and preprocess image
            image_raw = dlib.load_rgb_image(face_path)
            image_tensor = self.transform(image_raw).view(1, 3, 224, 224).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.fairface_model(image_tensor).cpu().numpy().squeeze()

            # Extract and normalize scores
            race_scores = self._softmax(outputs[:7])
            gender_scores = self._softmax(outputs[7:9])
            age_scores = self._softmax(outputs[9:18])

            # Get predictions
            race_pred = self.RACE_LABELS[np.argmax(race_scores)]
            gender_pred = self.GENDER_LABELS[np.argmax(gender_scores)]
            age_pred = self.AGE_LABELS[np.argmax(age_scores)]

            # Extract original image name
            original_image = os.path.basename(face_path).split("_face")[0]

            return DemographicPrediction(
                face_path=face_path,
                original_image=original_image,
                race=race_pred,
                gender=gender_pred,
                age=age_pred,
                race_scores=race_scores,
                gender_scores=gender_scores,
                age_scores=age_scores
            )

        except Exception as e:
            logger.error(f"Demographic prediction failed for {face_path}: {e}")
            return None

    def _mock_demographic_prediction(self, face_path: str) -> DemographicPrediction:
        """Create mock demographic predictions for testing."""
        # Random but deterministic predictions based on path hash
        path_hash = hash(face_path) % 1000

        race = self.RACE_LABELS[path_hash % len(self.RACE_LABELS)]
        gender = self.GENDER_LABELS[path_hash % len(self.GENDER_LABELS)]
        age = self.AGE_LABELS[path_hash % len(self.AGE_LABELS)]

        # Mock score arrays
        race_scores = np.random.rand(7)
        race_scores = race_scores / race_scores.sum()

        gender_scores = np.random.rand(2)
        gender_scores = gender_scores / gender_scores.sum()

        age_scores = np.random.rand(9)
        age_scores = age_scores / age_scores.sum()

        original_image = os.path.basename(face_path).split("_face")[0]

        return DemographicPrediction(
            face_path=face_path,
            original_image=original_image,
            race=race,
            gender=gender,
            age=age,
            race_scores=race_scores,
            gender_scores=gender_scores,
            age_scores=age_scores
        )

    @staticmethod
    def _softmax(x):
        """Compute softmax normalization."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def analyze_images(self, image_paths: List[str], output_dir: str = "face_analysis") -> pd.DataFrame:
        """Analyze a list of images for demographic characteristics.
        
        Args:
            image_paths: List of paths to images to analyze
            output_dir: Directory to save face crops and results
            
        Returns:
            DataFrame with demographic predictions

        """
        logger.info(f"Analyzing {len(image_paths)} images...")

        all_predictions = []

        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                logger.info(f"Processing image {i+1}/{len(image_paths)}")

            # Detect faces
            face_paths = self.detect_faces(image_path, os.path.join(output_dir, "faces"))

            # Predict demographics for each face
            for face_path in face_paths:
                prediction = self.predict_demographics(face_path)
                if prediction:
                    all_predictions.append(prediction)

        # Convert to DataFrame
        if not all_predictions:
            logger.warning("No faces detected in any images")
            return pd.DataFrame()

        records = []
        for pred in all_predictions:
            records.append({
                "face_path": pred.face_path,
                "original_image": pred.original_image,
                "race": pred.race,
                "gender": pred.gender,
                "age": pred.age,
                "race_scores": pred.race_scores,
                "gender_scores": pred.gender_scores,
                "age_scores": pred.age_scores
            })

        df = pd.DataFrame(records)

        # Save results
        results_path = os.path.join(output_dir, "demographic_predictions.csv")
        df.to_csv(results_path, index=False)
        logger.info(f"Demographic analysis saved to: {results_path}")

        return df

    def calculate_bias_w(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bias-W metrics (population-level bias).
        
        Args:
            df: DataFrame with demographic predictions
            
        Returns:
            DataFrame with Bias-W metrics for each attribute combination

        """
        bias_w_records = []

        for attribute_group in self.COMBINATIONS:
            attr_group_name = "_".join(attribute_group)

            # Calculate number of possible categories
            na = 1
            for attr in attribute_group:
                na *= len(self.ATTRIBUTES_DICT[attr])

            # Get frequency distribution
            if len(attribute_group) == 1:
                freq_w = df[attribute_group[0]].value_counts(normalize=True)
                all_combs = self.ATTRIBUTES_DICT[attribute_group[0]]
            else:
                freq_w = df.groupby(attribute_group).size() / len(df)
                if len(attribute_group) == 2:
                    all_combs = [(a, b) for a in self.ATTRIBUTES_DICT[attribute_group[0]]
                                        for b in self.ATTRIBUTES_DICT[attribute_group[1]]]
                else:  # 3 attributes
                    all_combs = [(a, b, c) for a in self.ATTRIBUTES_DICT[attribute_group[0]]
                                           for b in self.ATTRIBUTES_DICT[attribute_group[1]]
                                           for c in self.ATTRIBUTES_DICT[attribute_group[2]]]

            # Calculate Bias-W
            bias_w_sum = 0
            for comb in all_combs:
                if isinstance(comb, str):
                    freq_a_w = freq_w.get(comb, 0)
                else:
                    try:
                        freq_a_w = freq_w.loc[comb] if comb in freq_w.index else 0
                    except (KeyError, TypeError):
                        freq_a_w = 0

                bias_w_sum += (freq_a_w - (1 / na)) ** 2

            bias_w = np.sqrt((1 / na) * bias_w_sum)

            bias_w_records.append({
                "Attribute_Group": attr_group_name,
                "Bias_W": bias_w
            })

        return pd.DataFrame(bias_w_records)

    def calculate_bias_p(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bias-P metrics (per-image bias for multi-face images).
        
        Args:
            df: DataFrame with demographic predictions
            
        Returns:
            DataFrame with Bias-P metrics for each image and attribute combination

        """
        bias_p_records = []

        for attribute_group in self.COMBINATIONS:
            attr_group_name = "_".join(attribute_group)

            # Calculate number of possible categories
            na = 1
            for attr in attribute_group:
                na *= len(self.ATTRIBUTES_DICT[attr])

            # Get all possible combinations
            if len(attribute_group) == 1:
                all_combs = self.ATTRIBUTES_DICT[attribute_group[0]]
            elif len(attribute_group) == 2:
                all_combs = [(a, b) for a in self.ATTRIBUTES_DICT[attribute_group[0]]
                                    for b in self.ATTRIBUTES_DICT[attribute_group[1]]]
            else:  # 3 attributes
                all_combs = [(a, b, c) for a in self.ATTRIBUTES_DICT[attribute_group[0]]
                                       for b in self.ATTRIBUTES_DICT[attribute_group[1]]
                                       for c in self.ATTRIBUTES_DICT[attribute_group[2]]]

            # Calculate Bias-P for each image with multiple faces
            for image_name, group in df.groupby("original_image"):
                if len(group) > 1:  # Only multi-face images
                    # Get frequency distribution for this image
                    if len(attribute_group) == 1:
                        freq_p = group[attribute_group[0]].value_counts(normalize=True)
                    else:
                        freq_p = group.groupby(attribute_group).size() / len(group)

                    # Calculate Bias-P
                    bias_p_sum = 0
                    for comb in all_combs:
                        if isinstance(comb, str):
                            freq_a_p = freq_p.get(comb, 0)
                        else:
                            try:
                                freq_a_p = freq_p.loc[comb] if comb in freq_p.index else 0
                            except (KeyError, TypeError):
                                freq_a_p = 0

                        bias_p_sum += (freq_a_p - (1 / na)) ** 2

                    bias_p = np.sqrt((1 / na) * bias_p_sum)

                    bias_p_records.append({
                        "Original_Image": image_name,
                        "Attribute_Group": attr_group_name,
                        "Bias_P": bias_p
                    })

        return pd.DataFrame(bias_p_records)

    def calculate_ens(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Effective Number of Species (ENS) diversity metrics.
        
        Args:
            df: DataFrame with demographic predictions
            
        Returns:
            DataFrame with ENS metrics for each attribute combination

        """
        ens_records = []

        for attrs in self.COMBINATIONS:
            attrs = list(attrs)

            # Consider only rows where all attributes are present
            mask = df[attrs].notna().all(axis=1)
            sub = df.loc[mask, attrs].copy()

            denom = len(sub)
            if denom == 0:
                ens_records.append({
                    "Attribute": "+".join(attrs),
                    "ENS": 0.0
                })
                continue

            # Get all possible groups
            label_universes = [self.ATTRIBUTES_DICT[a] for a in attrs]
            all_groups = list(itertools.product(*label_universes))

            # Calculate proportions
            if len(attrs) == 1:
                counts = sub[attrs[0]].value_counts().to_dict()
                p_list = [counts.get(lab, 0) / denom for lab in self.ATTRIBUTES_DICT[attrs[0]]]
            else:
                counts = sub.value_counts().to_dict()
                p_list = []
                for grp in all_groups:
                    c = counts.get(tuple(grp), 0)
                    p_list.append(c / denom)

            # Compute ENS = exp(-Σ p_g ln p_g)
            sum_p_ln_p = 0.0
            for p_g in p_list:
                if p_g > 0:
                    sum_p_ln_p += p_g * np.log(p_g)

            ens_value = float(np.exp(-sum_p_ln_p))

            ens_records.append({
                "Attribute": "+".join(attrs),
                "ENS": ens_value
            })

        return pd.DataFrame(ens_records)

    def calculate_kl_divergence(self, df: pd.DataFrame, reference_distribution: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate KL divergence from reference distribution.
        
        Args:
            df: DataFrame with demographic predictions
            reference_distribution: Optional reference distributions
            
        Returns:
            DataFrame with KL divergence metrics

        """
        kl_records = []

        for attribute_group in self.COMBINATIONS:
            attr_group_name = "_".join(attribute_group)

            # Generated distribution
            if len(attribute_group) == 1:
                gen_dist = df[attribute_group[0]].value_counts(normalize=True)
            else:
                gen_dist = df.groupby(attribute_group).size() / len(df)

            # Reference distribution (uniform if not provided)
            if reference_distribution and attr_group_name in reference_distribution:
                ref_dist = pd.Series(reference_distribution[attr_group_name])
            else:
                # Uniform distribution
                if len(attribute_group) == 1:
                    labels = self.ATTRIBUTES_DICT[attribute_group[0]]
                else:
                    labels = list(itertools.product(*(self.ATTRIBUTES_DICT[attr] for attr in attribute_group)))
                ref_dist = pd.Series({label: 1/len(labels) for label in labels})

            # Align distributions
            all_labels = sorted(set(gen_dist.index) | set(ref_dist.index))
            gen_probs = np.array([gen_dist.get(label, 0) for label in all_labels], dtype=float)
            ref_probs = np.array([ref_dist.get(label, 0) for label in all_labels], dtype=float)

            # Clip to avoid log(0)
            eps = 1e-12
            gen_probs = np.clip(gen_probs, eps, 1)
            ref_probs = np.clip(ref_probs, eps, 1)

            # Calculate KL Divergence
            kl_value = np.sum(gen_probs * np.log(gen_probs / ref_probs))

            kl_records.append({
                "Attribute_Group": attr_group_name,
                "KL_Divergence": kl_value
            })

        return pd.DataFrame(kl_records)

    def evaluate_batch_results(
        self,
        original_images: List[str],
        enhanced_images: List[str],
        output_dir: str = "bias_evaluation"
    ) -> Tuple[BiasMetrics, BiasMetrics]:
        """Evaluate bias metrics for original vs enhanced images.
        
        Args:
            original_images: List of paths to original images
            enhanced_images: List of paths to enhanced images
            output_dir: Directory to save evaluation results
            
        Returns:
            Tuple of (original_metrics, enhanced_metrics)

        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Evaluating original images...")
        original_df = self.analyze_images(original_images, os.path.join(output_dir, "original"))

        logger.info("Evaluating enhanced images...")
        enhanced_df = self.analyze_images(enhanced_images, os.path.join(output_dir, "enhanced"))

        # Calculate metrics for original images
        original_metrics = BiasMetrics(
            bias_w_metrics=self.calculate_bias_w(original_df),
            bias_p_metrics=self.calculate_bias_p(original_df),
            ens_metrics=self.calculate_ens(original_df),
            kl_divergence_metrics=self.calculate_kl_divergence(original_df)
        )

        # Calculate metrics for enhanced images
        enhanced_metrics = BiasMetrics(
            bias_w_metrics=self.calculate_bias_w(enhanced_df),
            bias_p_metrics=self.calculate_bias_p(enhanced_df),
            ens_metrics=self.calculate_ens(enhanced_df),
            kl_divergence_metrics=self.calculate_kl_divergence(enhanced_df)
        )

        # Save all metrics
        self._save_metrics(original_metrics, os.path.join(output_dir, "original_metrics"))
        self._save_metrics(enhanced_metrics, os.path.join(output_dir, "enhanced_metrics"))

        # Save comparison
        self._save_comparison(original_metrics, enhanced_metrics, output_dir)

        return original_metrics, enhanced_metrics

    def _save_metrics(self, metrics: BiasMetrics, output_dir: str):
        """Save bias metrics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        metrics.bias_w_metrics.to_csv(os.path.join(output_dir, "bias_w_metrics.csv"), index=False)
        metrics.bias_p_metrics.to_csv(os.path.join(output_dir, "bias_p_metrics.csv"), index=False)
        metrics.ens_metrics.to_csv(os.path.join(output_dir, "ens_metrics.csv"), index=False)
        metrics.kl_divergence_metrics.to_csv(os.path.join(output_dir, "kl_divergence_metrics.csv"), index=False)

    def _save_comparison(self, original: BiasMetrics, enhanced: BiasMetrics, output_dir: str):
        """Save comparison of original vs enhanced metrics."""
        comparison_records = []

        # Compare Bias-W
        for _, row in original.bias_w_metrics.iterrows():
            attr_group = row["Attribute_Group"]
            original_value = row["Bias_W"]

            enhanced_row = enhanced.bias_w_metrics[
                enhanced.bias_w_metrics["Attribute_Group"] == attr_group
            ]
            enhanced_value = enhanced_row["Bias_W"].iloc[0] if not enhanced_row.empty else np.nan

            comparison_records.append({
                "Metric": "Bias_W",
                "Attribute_Group": attr_group,
                "Original": original_value,
                "Enhanced": enhanced_value,
                "Improvement": original_value - enhanced_value,
                "Improvement_Percent": ((original_value - enhanced_value) / original_value * 100) if original_value > 0 else 0
            })

        # Compare ENS
        for _, row in original.ens_metrics.iterrows():
            attr_group = row["Attribute"]
            original_value = row["ENS"]

            enhanced_row = enhanced.ens_metrics[
                enhanced.ens_metrics["Attribute"] == attr_group
            ]
            enhanced_value = enhanced_row["ENS"].iloc[0] if not enhanced_row.empty else np.nan

            comparison_records.append({
                "Metric": "ENS",
                "Attribute_Group": attr_group,
                "Original": original_value,
                "Enhanced": enhanced_value,
                "Improvement": enhanced_value - original_value,  # Higher ENS is better
                "Improvement_Percent": ((enhanced_value - original_value) / original_value * 100) if original_value > 0 else 0
            })

        # Compare KL Divergence
        for _, row in original.kl_divergence_metrics.iterrows():
            attr_group = row["Attribute_Group"]
            original_value = row["KL_Divergence"]

            enhanced_row = enhanced.kl_divergence_metrics[
                enhanced.kl_divergence_metrics["Attribute_Group"] == attr_group
            ]
            enhanced_value = enhanced_row["KL_Divergence"].iloc[0] if not enhanced_row.empty else np.nan

            comparison_records.append({
                "Metric": "KL_Divergence",
                "Attribute_Group": attr_group,
                "Original": original_value,
                "Enhanced": enhanced_value,
                "Improvement": original_value - enhanced_value,  # Lower KL is better
                "Improvement_Percent": ((original_value - enhanced_value) / original_value * 100) if original_value > 0 else 0
            })

        comparison_df = pd.DataFrame(comparison_records)
        comparison_df.to_csv(os.path.join(output_dir, "bias_comparison.csv"), index=False)

        logger.info(f"Bias evaluation complete. Results saved to: {output_dir}")
