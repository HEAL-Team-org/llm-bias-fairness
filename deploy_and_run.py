#!/usr/bin/env python3
"""Automated Deployment and Execution Script for LLM Bias-Fairness Pipeline.

This script automates the complete setup and execution process:
1. Clones the repository from GitHub
2. Creates a Python virtual environment
3. Installs all dependencies
4. Runs the parallel pipeline with environment-configured parameters

Environment Variables Required:
- GITHUB_REPO: Repository URL (default: https://github.com/pooriyasafaei/llm-bias-fairness.git)
- GITHUB_BRANCH: Branch to clone (default: code-refactor)
- DATA_FILE: Path to input CSV file with prompts (required)
- OPENAI_API_KEY: OpenAI API key for image generation (required)
- OPENAI_BASE_URL: Custom OpenAI endpoint (optional, default: https://api.openai.com/v1)
- NUM_WORKERS: Number of parallel workers (optional, default: 32)
- OUTPUT_DIR: Output directory for results (optional, default: results)
- USE_PROXYCHAINS: Set to "true" to use proxychains4 (optional, default: false)

Usage:
    export DATA_FILE="/path/to/prompts.csv"
    export OPENAI_API_KEY="sk-..."
    export NUM_WORKERS=32
    python3 deploy_and_run.py

Or with proxychains:
    export USE_PROXYCHAINS="true"
    python3 deploy_and_run.py
"""

import logging
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineDeployer:
    """Handles deployment and execution of the bias-fairness pipeline."""

    def __init__(self):
        """Initialize deployer with environment configuration."""
        # Repository settings
        self.repo_url = os.getenv(
            "GITHUB_REPO",
            "https://github.com/pooriyasafaei/llm-bias-fairness.git"
        )
        self.branch = os.getenv("GITHUB_BRANCH", "code-refactor")

        # Required parameters
        self.data_file = os.getenv("DATA_FILE")
        
        # Provider configuration (openai or azure)
        self.provider = os.getenv("OPENAI_PROVIDER", "openai").lower()
        
        # API credentials (provider-specific)
        if self.provider == "azure":
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            self.base_url = None  # Not used for Azure
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.azure_endpoint = None
            self.azure_deployment = None

        # Optional parameters
        self.num_workers = int(os.getenv("NUM_WORKERS", "32"))
        self.output_dir = os.getenv("OUTPUT_DIR", "results")
        self.use_proxychains = os.getenv("USE_PROXYCHAINS", "false").lower() == "true"

        # Deployment paths
        self.deploy_dir = Path.cwd() / "llm-bias-fairness-deployment"
        self.repo_dir = self.deploy_dir / "llm-bias-fairness"
        self.venv_dir = self.deploy_dir / "venv"

    def validate_environment(self):
        """Validate required environment variables."""
        logger.info("Validating environment configuration...")

        if not self.data_file:
            logger.error("DATA_FILE environment variable is required!")
            logger.error("Export it with: export DATA_FILE='/path/to/prompts.csv'")
            return False

        # Check if data file exists
        if not Path(self.data_file).exists():
            logger.error(f"Data file not found: {self.data_file}")
            return False

        # Validate provider-specific credentials
        if self.provider == "azure":
            if not self.api_key:
                logger.error("AZURE_OPENAI_API_KEY environment variable is required for Azure mode!")
                logger.error("Export it with: export AZURE_OPENAI_API_KEY='your-key'")
                return False
            if not self.azure_endpoint:
                logger.error("AZURE_OPENAI_ENDPOINT environment variable is required for Azure mode!")
                logger.error("Export it with: export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
                return False
            logger.info(f"✓ Provider: Azure OpenAI")
            logger.info(f"✓ Azure endpoint: {self.azure_endpoint}")
            if self.azure_deployment:
                logger.info(f"✓ Azure deployment: {self.azure_deployment}")
        else:
            if not self.api_key:
                logger.error("OPENAI_API_KEY environment variable is required!")
                logger.error("Export it with: export OPENAI_API_KEY='sk-...'")
                return False
            logger.info(f"✓ Provider: Standard OpenAI")
            logger.info(f"✓ Base URL: {self.base_url}")

        logger.info(f"✓ Repository: {self.repo_url}")
        logger.info(f"✓ Branch: {self.branch}")
        logger.info(f"✓ Data file: {self.data_file}")
        logger.info(f"✓ API key: {self.api_key[:10]}...{self.api_key[-5:]}")
        logger.info(f"✓ Workers: {self.num_workers}")
        logger.info(f"✓ Output dir: {self.output_dir}")
        logger.info(f"✓ Use proxychains: {self.use_proxychains}")

        return True

    def check_prerequisites(self):
        """Check if required system tools are installed."""
        logger.info("Checking prerequisites...")

        # Check git
        if not shutil.which("git"):
            logger.error("Git is not installed. Please install git first.")
            return False
        logger.info("✓ Git found")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ is required")
            return False
        logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

        # Check proxychains if needed
        if self.use_proxychains:
            if not shutil.which("proxychains4"):
                logger.error("proxychains4 not found but USE_PROXYCHAINS=true")
                return False
            logger.info("✓ proxychains4 found")

        return True

    def clone_repository(self):
        """Clone the GitHub repository."""
        logger.info("=" * 70)
        logger.info("CLONING REPOSITORY")
        logger.info("=" * 70)

        # Clean up existing deployment
        if self.deploy_dir.exists():
            logger.info(f"Removing existing deployment: {self.deploy_dir}")
            shutil.rmtree(self.deploy_dir)

        # Create deployment directory
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created deployment directory: {self.deploy_dir}")

        # Clone repository
        logger.info(f"Cloning {self.repo_url} (branch: {self.branch})...")
        try:
            subprocess.run(
                ["git", "clone", "-b", self.branch, self.repo_url, str(self.repo_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✓ Repository cloned to: {self.repo_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            return False

    def create_virtual_environment(self):
        """Create Python virtual environment."""
        logger.info("=" * 70)
        logger.info("CREATING VIRTUAL ENVIRONMENT")
        logger.info("=" * 70)

        try:
            logger.info(f"Creating virtual environment at: {self.venv_dir}")
            venv.create(self.venv_dir, with_pip=True)
            logger.info("✓ Virtual environment created")
            return True
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False

    def install_dependencies(self):
        """Install Python dependencies."""
        logger.info("=" * 70)
        logger.info("INSTALLING DEPENDENCIES")
        logger.info("=" * 70)

        # Get pip path
        if sys.platform == "win32":
            pip_path = self.venv_dir / "Scripts" / "pip"
        else:
            pip_path = self.venv_dir / "bin" / "pip"

        requirements_file = self.repo_dir / "requirements.txt"

        if not requirements_file.exists():
            logger.error(f"requirements.txt not found: {requirements_file}")
            return False

        try:
            # Upgrade pip
            logger.info("Upgrading pip...")
            subprocess.run(
                [str(pip_path), "install", "--upgrade", "pip"],
                check=True,
                capture_output=True
            )

            # Install requirements
            logger.info("Installing requirements...")
            subprocess.run(
                [str(pip_path), "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True
            )
            logger.info("✓ Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def create_config_file(self):
        """Create configuration file with environment settings."""
        logger.info("=" * 70)
        logger.info("CREATING CONFIGURATION")
        logger.info("=" * 70)

        config_dir = self.repo_dir / "config"
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / "config.yaml"

        # Generate provider-specific configuration
        if self.provider == "azure":
            config_content = f"""# Auto-generated configuration
# Generated by deploy_and_run.py
# Provider: Azure OpenAI

openai:
  provider: "azure"
  api_key: "{self.api_key}"

  # Azure-specific settings
  azure_endpoint: "{self.azure_endpoint}"
  azure_deployment: "{self.azure_deployment or 'gpt-4'}"
  api_version: "2024-02-15-preview"

  # Model names (for Azure, these are deployment names)
  embedding_model: "text-embedding-3-large"
  chat_model: "gpt-5"
  image_model: "gpt-image-1"

  # API settings
  max_retries: 3
  timeout: 120

# Image Generation Configuration
image_generation:
  default_size: "1024x1024"
  default_quality: "standard"
  default_style: "vivid"
"""
        else:
            config_content = f"""# Auto-generated configuration
# Generated by deploy_and_run.py
# Provider: Standard OpenAI

openai:
  provider: "openai"
  api_key: "{self.api_key}"
  base_url: "{self.base_url}"
  
  # Model names
  embedding_model: "text-embedding-3-large"
  chat_model: "gpt-5"
  image_model: "gpt-image-1"
  
  # API settings
  max_retries: 3
  timeout: 120

# Image Generation Configuration
image_generation:
  default_size: "1024x1024"
  default_quality: "standard"
  default_style: "vivid"
"""

        try:
            config_file.write_text(config_content)
            logger.info(f"✓ Configuration created: {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create config: {e}")
            return False

    def run_pipeline(self):
        """Run the parallel pipeline."""
        logger.info("=" * 70)
        logger.info("RUNNING PARALLEL PIPELINE")
        logger.info("=" * 70)

        # Get Python path
        if sys.platform == "win32":
            python_path = self.venv_dir / "Scripts" / "python"
        else:
            python_path = self.venv_dir / "bin" / "python3"

        pipeline_script = self.repo_dir / "run_pipeline_parallel.py"

        if not pipeline_script.exists():
            logger.error(f"Pipeline script not found: {pipeline_script}")
            return False

        # Copy data file to repo directory
        data_file_name = Path(self.data_file).name
        local_data_file = self.repo_dir / data_file_name
        shutil.copy2(self.data_file, local_data_file)
        logger.info(f"Copied data file to: {local_data_file}")

        # Prepare command
        cmd = [
            str(python_path),
            str(pipeline_script),
            data_file_name,
            "--workers", str(self.num_workers),
            "--output-dir", self.output_dir
        ]

        # Add proxychains if needed
        if self.use_proxychains:
            cmd = ["proxychains4"] + cmd

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.repo_dir}")
        logger.info("\nStarting pipeline execution...\n")

        try:
            # Set environment variables
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = self.api_key

            # Run pipeline
            process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output in real-time
            for line in process.stdout:
                print(line, end="")

            process.wait()

            if process.returncode == 0:
                logger.info("\n✓ Pipeline completed successfully!")
                return True
            logger.error(f"\n✗ Pipeline failed with exit code: {process.returncode}")
            return False

        except Exception as e:
            logger.error(f"Failed to run pipeline: {e}")
            return False

    def get_results_info(self):
        """Display information about results location."""
        logger.info("=" * 70)
        logger.info("RESULTS LOCATION")
        logger.info("=" * 70)

        results_dir = self.repo_dir / self.output_dir

        if results_dir.exists():
            logger.info(f"Results directory: {results_dir}")
            logger.info(f"  - Images: {results_dir / 'generated_images'}")
            logger.info(f"  - Complete results: {results_dir / 'complete_pipeline_results.json'}")

            # Count files
            if (results_dir / "generated_images").exists():
                num_images = len(list((results_dir / "generated_images").glob("*.png")))
                logger.info(f"  - Total images generated: {num_images}")
        else:
            logger.warning(f"Results directory not found: {results_dir}")

    def deploy_and_run(self):
        """Execute complete deployment and run process."""
        logger.info("=" * 70)
        logger.info("LLM BIAS-FAIRNESS PIPELINE - AUTOMATED DEPLOYMENT")
        logger.info("=" * 70)
        logger.info("")

        # Validate environment
        if not self.validate_environment():
            return False

        # Check prerequisites
        if not self.check_prerequisites():
            return False

        # Clone repository
        if not self.clone_repository():
            return False

        # Create virtual environment
        if not self.create_virtual_environment():
            return False

        # Install dependencies
        if not self.install_dependencies():
            return False

        # Create config
        if not self.create_config_file():
            return False

        # Run pipeline
        if not self.run_pipeline():
            return False

        # Show results
        self.get_results_info()

        logger.info("\n" + "=" * 70)
        logger.info("DEPLOYMENT AND EXECUTION COMPLETE!")
        logger.info("=" * 70)

        return True


def main():
    """Main entry point."""
    deployer = PipelineDeployer()

    try:
        success = deployer.deploy_and_run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
