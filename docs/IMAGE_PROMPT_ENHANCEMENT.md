# Image Prompt Enhancement for Diversity and Bias Mitigation

## 🎨 Overview

The **GraphRAG Image Prompt Enhancement System** is a specialized tool that leverages your knowledge graphs containing bias/stereotype data and cultural values to enhance image generation prompts. This system promotes diversity, cultural sensitivity, and bias mitigation in AI-generated images.

## 🎯 Problem Solved

When generating images with AI tools like DALL-E, Midjourney, or Stable Diffusion, prompts often result in biased or non-diverse outputs due to:
- **Training data biases** in the image generation models
- **Lack of explicit diversity** in user prompts  
- **Cultural insensitivity** or stereotypical representations
- **Missing accessibility considerations** for people with disabilities

## 🛠️ How It Works

### 1. Input Processing
- User provides an image generation prompt (e.g., "a doctor examining a patient")
- System accepts input via command line or interactive mode

### 2. Knowledge Retrieval
- **Bias Graph**: Retrieves top-K relevant bias/stereotype triples to identify potential harmful representations
- **Cultural Values Graph**: Retrieves top-K cultural value triples to promote inclusive practices
- Uses semantic similarity (with OpenAI embeddings) or substring matching as fallback

### 3. LLM Enhancement
- Constructs a comprehensive prompt for the LLM containing:
  - Original user prompt
  - Retrieved bias information (to avoid)
  - Retrieved cultural values (to incorporate)
  - Specific instructions for diversity enhancement
- Uses GPT-4 to generate an enhanced, inclusive prompt

### 4. Output Generation
- Provides the enhanced prompt with explicit diversity elements
- Includes explanation of changes made
- Lists specific diversity aspects incorporated

## 📊 Example Enhancement

### Original Prompt:
```
"a doctor examining a patient"
```

### Enhanced Prompt:
```
"a doctor examining a patient, featuring people of diverse ages including young adults, middle-aged individuals, and seniors, representing various ethnicities including Asian, African, Latino, Middle Eastern, and European backgrounds, with different skin tones and physical appearances, wearing culturally diverse clothing and accessories, in an inclusive environment that celebrates global diversity, with both men and women and non-binary individuals, including people with visible and invisible disabilities, showcasing different socioeconomic backgrounds through varied but respectful styling, with authentic cultural elements like traditional patterns, diverse architectural styles, and inclusive symbols that promote unity and respect across all communities"
```

### Diversity Elements Added:
- **Age diversity**: Multiple generations represented
- **Ethnic diversity**: Asian, African, Latino, Middle Eastern, European backgrounds
- **Gender inclusivity**: Men, women, non-binary individuals
- **Disability representation**: Visible and invisible disabilities
- **Cultural authenticity**: Traditional clothing, patterns, architectural diversity
- **Socioeconomic representation**: Varied but respectful styling
- **Inclusive environment**: Globally-inspired settings

## 🚀 Usage Examples

### Interactive Mode
```bash
python enhance_prompt.py
# System prompts for input with helpful examples
```

### Command Line Mode
```bash
# Basic usage
python enhance_prompt.py -p "students in a classroom"

# Custom retrieval parameters
python enhance_prompt.py -p "engineers working on a project" --bias-top-k 15 --cultural-top-k 20

# Different cache file
python enhance_prompt.py -p "family having dinner" --cache-file custom_embeddings.pkl
```

## 📈 Performance Metrics

Based on testing with your datasets:

### Data Processing
- **Bias graph**: 51,301 triples (13,705 nodes)
- **Cultural values graph**: 6,469 triples (5,693 nodes) from 15 countries
- **Embedding cache**: 13,737+ cached embeddings for efficiency

### Retrieval Results (Example: "black" prompt)
- **Bias triples found**: 7,701 relevant triples
- **Cultural triples found**: 21 relevant triples
- **Processing time**: < 30 seconds including data loading

### Enhancement Quality
- ✅ **Bias mitigation**: Successfully identifies and avoids harmful stereotypes
- ✅ **Cultural sensitivity**: Incorporates diverse cultural values and practices  
- ✅ **Explicit diversity**: Adds specific age, ethnic, gender, disability representation
- ✅ **Prompt preservation**: Maintains original concept while enhancing inclusivity

## 🔧 Technical Implementation

### Key Components
1. **`enhance_prompt.py`**: Main script with CLI interface
2. **GraphRAG system**: Leverages existing modular architecture
3. **Data parsers**: Handles both CSV bias data and text cultural values
4. **LLM integration**: Uses OpenAI GPT models for enhancement generation
5. **Caching system**: Persistent embedding storage for efficiency

### Fallback Mechanisms
- **No OpenAI API**: Uses mock enhancement with diversity template
- **No vector embeddings**: Falls back to substring matching
- **Missing data files**: Graceful error handling with informative messages

## 🎯 Use Cases

### Content Creation
- **Marketing materials**: Ensure diverse representation in advertising
- **Educational content**: Create inclusive imagery for learning materials
- **Stock photography**: Generate diverse, representative images
- **Social media**: Promote inclusive visual content

### AI Ethics & Research
- **Bias studies**: Analyze and mitigate bias in image generation
- **Cultural research**: Study cross-cultural representation in AI
- **Diversity metrics**: Measure inclusivity in generated content
- **Educational tools**: Teach about bias and diversity in AI systems

### Creative Applications
- **Art projects**: Create culturally sensitive artistic works
- **Game development**: Design diverse characters and environments
- **Film/TV**: Ensure inclusive representation in visual media
- **Publishing**: Generate diverse illustrations for books and articles

## 🌍 Impact

This system addresses critical issues in AI-generated imagery:

### Bias Mitigation
- **Reduces harmful stereotypes** by explicitly identifying and avoiding biased representations
- **Promotes positive representation** of marginalized groups
- **Challenges default assumptions** about occupations, activities, and social roles

### Cultural Sensitivity  
- **Incorporates global perspectives** from 15+ countries in the cultural values dataset
- **Respects cultural practices** and traditions in visual representations
- **Promotes cross-cultural understanding** through inclusive imagery

### Accessibility & Inclusion
- **Represents disabilities** both visible and invisible
- **Includes diverse ages** from children to seniors
- **Promotes gender inclusivity** beyond binary representations
- **Considers socioeconomic diversity** in respectful ways

## 🔮 Future Enhancements

### Additional Data Sources
- **Accessibility guidelines** for better disability representation
- **Indigenous knowledge** systems and cultural practices
- **LGBTQ+ representation** guidelines and cultural sensitivity
- **Regional diversity** data for more specific cultural contexts

### Enhanced AI Integration
- **Multi-modal analysis** of generated images for bias detection
- **Real-time feedback** on prompt effectiveness
- **Automated diversity scoring** of enhanced prompts
- **Integration with image generation APIs** for end-to-end workflow

### User Experience
- **Web interface** for easier prompt enhancement
- **Batch processing** for multiple prompts
- **Custom bias/cultural datasets** for specific use cases
- **A/B testing framework** for prompt effectiveness

## 📝 Conclusion

The GraphRAG Image Prompt Enhancement System represents a significant step forward in creating more inclusive, diverse, and culturally sensitive AI-generated imagery. By leveraging structured knowledge about biases and cultural values, it helps users create prompts that promote positive representation while avoiding harmful stereotypes.

This tool is particularly valuable for content creators, researchers, educators, and anyone committed to creating more inclusive visual content in the age of AI-generated imagery.
