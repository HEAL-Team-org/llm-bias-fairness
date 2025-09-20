# Complete Prompt Enhancement and Image Generation Pipeline

This project provides a comprehensive pipeline for bias-aware prompt enhancement and image generation. It combines three knowledge sources (StereoSet, CultureBank, and GraphRAG) to enhance prompts for better bias mitigation and cultural diversity, then generates before/after images for comparison.

## 🚀 Features

- **Dual-Pipeline Enhancement**: Combines stereotype detection, cultural diversity, and knowledge graph enhancement
- **Bias Scoring**: Quantitative bias mitigation scoring (0-100)
- **Diversity Scoring**: Cultural diversity enhancement scoring (0-100)
- **Generic Image Generation**: Supports DALL-E 3 with extensible interface for other models
- **Batch Processing**: Process CSV files with multiple prompts
- **Trackable Results**: All images and metadata saved with clear naming conventions

## 📁 Project Structure

```
llm-bias-fairness/
├── enhance_prompt_dual_pipeline.py    # Main dual-pipeline enhancement system
├── batch_processor.py                 # Batch CSV processing pipeline
├── test_batch_pipeline.py            # Test script for the complete pipeline
├── test_prompts.csv                  # Sample prompts for testing
├── src/
│   ├── image_generator.py            # Generic image generation interface
│   ├── stereoset_rag.py             # Stereotype detection (4,229 examples)
│   ├── diversity_rag.py             # Cultural diversity (22,990 behaviors)
│   └── graphrag.py                  # Knowledge graph (51,301+ triples)
└── data/
    ├── biases/                      # Bias detection datasets
    └── cultural_values/             # Cultural diversity datasets
```

## 🔧 Installation

1. **Install Python Dependencies**:
```bash
pip install pandas numpy scikit-learn openai pillow requests
```

2. **Set OpenAI API Key** (for DALL-E 3):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Verify Installation**:
```bash
python test_batch_pipeline.py
```

## 🚀 Quick Start

### 1. Test with Sample Data

Run the test script with mock image generation:
```bash
python test_batch_pipeline.py
```

This will:
- Process the first 3 prompts from `test_prompts.csv`
- Use mock image generation (no API calls)
- Show you the complete pipeline flow

### 2. Process Your Own CSV File

```bash
python batch_processor.py your_file.csv --prompt-column "your_prompt_column"
```

### 3. Use Real DALL-E 3 Image Generation

```bash
python batch_processor.py test_prompts.csv --image-generator dalle3
```

## 📊 Input/Output Format

### Input CSV Format
Your CSV file should have at least one column with prompts:

```csv
prompt,description,category
"A professional photo of a person working in an office","Basic workplace photo","workplace"
"A doctor examining a patient in a hospital","Medical professional scenario","healthcare"
```

### Output Files

The pipeline generates several output files:

1. **Enhanced CSV**: Original data + enhancement results
   - `enhanced_prompt`: Bias-aware and culturally diverse version
   - `bias_score`: Bias mitigation score (0-100)
   - `diversity_score`: Cultural diversity score (0-100)
   - `original_image_path`: Path to original image
   - `enhanced_image_path`: Path to enhanced image

2. **Detailed JSON**: Complete metadata including:
   - Enhancement reasoning and sources
   - Image generation metadata
   - Processing times and errors

3. **Generated Images**: 
   - `row_0001_original.png`: Image from original prompt
   - `row_0001_enhanced.png`: Image from enhanced prompt

## 🎯 Command Line Options

```bash
python batch_processor.py INPUT_CSV [OPTIONS]

Arguments:
  INPUT_CSV                    Path to input CSV file

Options:
  --prompt-column TEXT         Name of column containing prompts [default: prompt]
  --output-csv TEXT           Path for output CSV file [auto-generated]
  --start-row INTEGER         Row to start processing from [default: 0]
  --max-rows INTEGER          Maximum number of rows to process
  --image-generator TEXT      Type of image generator: dalle3, mock [default: dalle3]
  --output-dir TEXT           Directory for batch results [default: batch_results]
  --image-dir TEXT            Directory for generated images [default: generated_images]
  --no-save-intermediate      Don't save intermediate results
```

## 🔬 Enhancement Pipeline Details

### 1. Dual-Pipeline Enhancement System

The system uses three complementary knowledge sources:

- **StereoSet RAG**: 4,229 academic bias examples for stereotype detection
- **CultureBank RAG**: 22,990 cultural behaviors from social media for diversity
- **GraphRAG**: 51,301+ knowledge triples from bias research and cultural datasets

### 2. Sequential Enhancement Process

1. **Initial Enhancement**: Basic bias mitigation and diversity improvement
2. **Bias Scoring**: Quantitative assessment of bias reduction
3. **Diversity Scoring**: Cultural representation improvement measurement
4. **Threshold-Based Iteration**: Re-enhance if scores don't meet thresholds

### 3. Scoring System

- **Bias Score (0-100)**: Higher = better bias mitigation
- **Diversity Score (0-100)**: Higher = more culturally inclusive

## 🖼️ Image Generation

### Supported Generators

1. **DALL-E 3** (`--image-generator dalle3`):
   - High-quality image generation
   - Requires OpenAI API key
   - Automatic prompt revision by OpenAI

2. **Mock Generator** (`--image-generator mock`):
   - For testing without API costs
   - Generates simple placeholder images

### Adding New Image Generators

The system uses a generic interface. To add new generators:

```python
from src.image_generator import BaseImageGenerator

class YourGenerator(BaseImageGenerator):
    def generate_image(self, prompt, filename=None, **kwargs):
        # Your implementation
        return saved_path, metadata
```

## 📈 Example Results

### Original Prompt:
"A professional photo of a person working in an office"

### Enhanced Prompt:
"A professional photo showcasing a diverse individual working in a modern, inclusive office environment, representing the global workforce with attention to equal representation and accessibility"

### Scores:
- **Bias Score**: 85/100 (significant bias reduction)
- **Diversity Score**: 78/100 (improved cultural inclusivity)

## 🔧 Configuration

### Default Thresholds

```python
bias_threshold = 60      # Minimum bias score to accept
diversity_threshold = 40 # Minimum diversity score to accept
```

### Image Generation Settings

```python
# DALL-E 3 settings
model = "dall-e-3"
size = "1024x1024"         # Options: 1024x1024, 1792x1024, 1024x1792
quality = "standard"       # Options: standard, hd
style = "vivid"           # Options: vivid, natural
```

## 🚨 Troubleshooting

### Common Issues

1. **"OpenAI package not installed"**:
   ```bash
   pip install openai
   ```

2. **"Column 'prompt' not found"**:
   - Check your CSV column names
   - Use `--prompt-column "your_column_name"`

3. **"Failed to initialize image generator"**:
   - For DALL-E 3: Set `OPENAI_API_KEY` environment variable
   - For testing: Use `--image-generator mock`

4. **High API costs**:
   - Start with `--max-rows 5` for testing
   - Use `--image-generator mock` for development

### Performance Tips

- Use `--max-rows` to process in batches
- Enable `--no-save-intermediate` for faster processing
- Process during off-peak hours for better API response times

## 📝 Development

### Running Tests

```bash
# Test with mock image generation
python test_batch_pipeline.py

# Test with real DALL-E 3 (requires API key)
OPENAI_API_KEY="your-key" python test_batch_pipeline.py
```

### Adding New Knowledge Sources

1. Create a new RAG class in `src/`
2. Add it to the dual pipeline in `enhance_prompt_dual_pipeline.py`
3. Update scoring functions to include new source

## 📄 License

This project is for academic research purposes. Please cite appropriately if used in publications.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test script output
3. Open an issue with detailed error logs
