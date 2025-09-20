# Intent Classification with BERT: 95% Accuracy for IT Services

A fine-tuned BERT model achieving 95% accuracy on IT services intent classification. This project demonstrates how to effectively classify customer queries into predefined intents using state-of-the-art transformer models.

## 🎯 Problem Statement

Customer service chatbots need to accurately understand user intents to provide relevant responses. This project solves intent classification for IT services with the following categories:

- **Requirement Submission**: Technical requirements and specifications
- **General Query**: General questions about services
- **Contact Details**: Requests for contact information
- **Feedback Submission**: User feedback and reviews
- **Appreciation**: Thank you messages and compliments
- **Greeting**: Hello, hi, and conversation starters
- **Job Application Submission**: Applications for job positions

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/monology-io/intent-classification-bert.git
cd intent-classification-bert
pip install -r requirements.txt
```

### Training Your Model

#### Option 1: Google Colab (FREE GPU - Recommended)

**Train completely FREE with Google Colab's GPU:**

1. **Open Google Colab** → [colab.research.google.com](https://colab.research.google.com)
2. **Create new notebook** → File > New notebook
3. **Enable GPU** → Runtime > Change runtime type > GPU
4. **Copy & paste** the complete script from [`notebooks/colab_training.py`](https://raw.githubusercontent.com/monology-io/intent-classification-bert/main/notebooks/colab_training.py) into a code cell
5. **Run the cell** and follow the interactive prompts

**What the Colab script does:**

- 🔧 Installs all required packages automatically (It will ask you to restart the runtime, please restart it and run the cell again)
- 📊 Ask you to upload your own dataset (sample data is provided: /data/sample_data.csv)
- 🏃‍♂️ Trains the model with progress monitoring
- 📈 Shows detailed performance metrics and confusion matrix
- 🧪 Tests the model with sample queries interactively
- 💾 Saves everything to Google Colab files (It is available in Colab files, please download it before session end)

**Colab Benefits & Limitations:**

- ✅ Free GPU access (Tesla T4), no setup required
- ⚠️ 12-hour session limit, save to Drive to prevent data loss
- ⏱️ Training time: 3-5 min (sample data), 10-15 min (1K samples)

#### Option 2: Local Training

```bash
python train.py
```

**Requirements**: GPU recommended, 8GB+ RAM, Python 3.8+

**Important**: The provided sample data is minimal. For production-quality results, prepare a dataset with 1,000+ samples per intent.

#### Making Predictions

```python
from inference import IntentClassifier

classifier = IntentClassifier("./intent_model")
result = classifier.predict("Hello, I need help with my server setup")
print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.2f})")
```

## 📊 Performance

- **Accuracy**: 95.2% _(achieved with 70,000+ training samples)_
- **Training Time**: ~15 minutes on GPU _(for full dataset)_
- **Model Size**: 438MB
- **Inference Speed**: ~50ms per query

**Note**: Performance metrics above are based on our production dataset with 10,000+ samples per intent. Results with the provided sample data will be significantly lower.

## 📁 Dataset Format

Your CSV file should have two columns:

```csv
query,intent
Hi there, how can I get started?,Greeting
I need to submit a technical requirement,Requirement Submission
What are your contact details?,Contact Details
```

## ⚠️ Important Note on Dataset

**The `data/sample_data.csv` file contains only limited sample data for demonstration and getting started purposes.**

To achieve the claimed **95% accuracy**, you need a substantial dataset:

- **Minimum recommended**: 1,000+ samples per intent
- **Our production model**: Trained on 10,000+ samples per intent
- **Total training data**: 70,000+ labeled queries across all 7 intents

The sample data provided here will help you:

- ✅ Test the training pipeline
- ✅ Understand the data format
- ✅ Validate your setup
- ❌ **Will NOT achieve 95% accuracy**

For production use, you'll need to collect and label a comprehensive dataset with diverse query variations, different phrasings, and real-world examples for each intent category.

## 🔧 Configuration

Modify `config/intent_mapping.json` to add or change intent categories:

```json
{
  "Requirement Submission": 0,
  "General Query": 1,
  "Contact Details": 2,
  "Feedback Submission": 3,
  "Appreciation": 4,
  "Greeting": 5,
  "Job Application Submission": 6
}
```

## 📈 Training Configuration

Adjust training parameters in `config/training_config.yaml`:

```yaml
model_name: "bert-base-uncased"
learning_rate: 2e-5
batch_size: 8
num_epochs: 3
max_length: 128
eval_strategy: "epoch"
```

## 📋 Requirements

### For Google Colab (Recommended)

- ✅ No installation required - everything runs in browser
- ✅ Free GPU access (Tesla T4)
- ✅ 12GB RAM provided
- ⚠️ 12-hour session limits (save to Drive)

### For Local Setup

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.21+
- CUDA compatible GPU (recommended)
- 8GB RAM minimum
- **Large dataset**: 1,000+ samples per intent for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat(intent-classification-bert): add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers team for the excellent library
- Google for BERT architecture
- The open-source ML community

## 📞 Support & Resources

If you have questions or need help:

- Open an issue on GitHub
- Email: contact@monology.io
- LinkedIn: [Monology](https://www.linkedin.com/company/monology-io)

## 🌐 About This Project

This intent classification system was developed as part of the **Monology** platform - an AI chatbot builder that uses multi-agent workflows. If you're looking for a production-ready solution with visual workflow builders, pre-trained models, and enterprise features, check out [Monology's IT Services Hub](https://monology.io/it-services-hub).

For more information about Monology: [monology.io](https://monology.io)

## 🔗 Related Resources

- [Monology Platform](https://monology.io) - Build AI chatbots with visual workflows
- [IT Services Hub](https://monology.io/it-services-hub) - Pre-configured IT services workflow

---

⭐ If this project helped you, please give it a star on GitHub!
