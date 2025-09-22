# PathOCL: Path-Based Prompt Augmentation for OCL Generation with Open-Source LLMs

This repository contains an unofficial implementation of **PathOCL**, a path-based prompt augmentation approach for generating Object Constraint Language (OCL) constraints from natural language specifications. The original research paper proposed using GPT-4, but this implementation extends the approach to various state-of-the-art open-source language models.

> **Paper Reference**: PathOCL: Path-Based Prompt Augmentation for OCL Generation with GPT-4  
> **DOI**: [https://doi.org/10.1145/3650105.3652290](https://doi.org/10.1145/3650105.3652290)

## Overview

PathOCL transforms natural language specifications into valid OCL constraints by:

1. **Linguistic Processing**: Extracting UML elements from natural language using spaCy
2. **Path-Based Augmentation**: Generating relevant UML class paths and their properties
3. **Similarity Ranking**: Using Jaccard and cosine similarity to select the most relevant paths
4. **LLM Generation**: Generating OCL constraints using augmented prompts with selected UML context

## Supported LLM Models

This implementation supports the following open-source models:

- **Llama 3 8B** (`llama3-8b`)
- **Qwen 3 8B** (`qwen3-8b`) 
- **Phi 4 14B** (`phi4-14b`)
- **Gemma 7B** (`gemma-7b`)
- **Ministral 8B** (`ministral-8b`)

## Project Structure
```
pathocl_project/
├── pathocl_dataset/
│ ├── UML/
│ │ ├── Airport/
│ │ │ └── Airport.puml
│ │ └── ... (other domains)
│ └── specification.json
├── PathOCL.py
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/HamzaGiTX786/PathOCL.git
cd PathOCL
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Set up HuggingFace token**:
- Obtain your HuggingFace access token from `huggingface.co/settings/tokens`
- You will need this token to access the LLM models

---

## Usage

### Basic Command
```bash
python PathOCL.py \
    --domain <domain_name> \
    --spec "<natural_language_specification>" \
    --spec-file <path_to_specifications.json> \
    --llm <model_name> \
    --hf-token <your_huggingface_token> \
    [--top-k <number_of_paths>]
```

### Parameters

--domain: Domain name (e.g., "Airport", "Tournament")
--spec: Natural language specification text (in quotes)
--spec-file: Path to the specifications JSON file
--llm: LLM model to use (llama3-8b, qwen3-8b, phi4-14b, gemma-7b, ministral-8b)
--hf-token: Your HuggingFace access token
--top-k: Number of top paths to consider (default: 3)

### Example
```bash
python pathocl_implementation.py \
    --domain Airport \
    --spec "All flight objects must have a duration attribute that is less than four" \
    --spec-file ./pathocl_dataset/specification.json \
    --llm llama3-8b \
    --hf-token hf_xxxxxxxxxxxxxxxx
```

## Methodology

1. Linguistic Processing
- Uses spaCy to extract nouns and adjectives from natural language specifications
- Applies mapping rules to identify UML element relationships

2. Simple Path Generation
- Parses UML diagrams to extract classes, attributes, operations, and associations
- Generates simple paths between classes in the UML model

3. Similarity Calculation
- **Jaccard Similarity**: Measures overlap between specification elements and UML properties
- **Cosine Similarity**: Uses sentence transformers for semantic similarity

4. Prompt Augmentation
- Constructs detailed prompts containing:
    - Natural language specification
    - UML class properties (attributes, operations, associations)
    - Relevant path context

5.  LLM Generation
- Uses the selected open-source LLM to generate OCL constraints
- Maintains model-specific input formats (chat templates, special tokens)

### Output
**LLM Response**: The generated OCL constraint wrapped in <OCL> tags

Example Output: 
```bash
Generated Prompt:
-- OCL specification 
All flight objects must have a duration attribute that is less than four

-- UML properties of class Flight 
{ 
    "attributes": 
    [ 
        { "duration": "Integer" }
    ], 
    ...
}

-- OCL constraint

LLM Output:
<OCL>
context Flight inv: self.duration < 4
</OCL>
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.