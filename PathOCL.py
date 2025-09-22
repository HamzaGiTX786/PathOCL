from collections import defaultdict
import spacy
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import login
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import json
import argparse
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def linguistic_process(sentence):
    doc = nlp(sentence)
    nouns = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    adjs  = [token.lemma_.lower() for token in doc if token.pos_ == "ADJ"]
    return nouns, adjs, doc

def apply_mapping_rules(doc):
    elements = []

    for token in doc:
        if token.pos_ == "NUM":  # skip numbers (constants, not UML elements)
            continue

        # Rule: IsPropertyOfSign ("name of X" → X.name)
        if token.text.lower() == "name" and token.i + 2 < len(doc):
            if doc[token.i+1].text.lower() == "of":
                head = doc[token.i+2]
                if head.pos_ != "NUM":
                    elements.append(f"{head.lemma_.lower()}.name")

        # Rule: PossessiveDeterminer ("X's Y" → X.Y)
        if token.dep_ == "poss" and token.head.pos_ != "NUM":
            elements.append(f"{token.head.lemma_.lower()}.{token.lemma_.lower()}")

        # Rule: PrefixElement ("flight duration" → flight.duration)
        if token.dep_ == "compound" and token.head.pos_ != "NUM":
            elements.append(f"{token.head.lemma_.lower()}.{token.lemma_.lower()}")

        # Rule: TransitiveVerb ("flight has passenger")
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            subj = [t for t in token.lefts if t.dep_ in ("nsubj", "nsubjpass")]
            obj = [t for t in token.rights if t.dep_ in ("dobj", "attr", "pobj")]
            if subj and obj and obj[0].pos_ != "NUM":
                elements.append(f"{subj[0].lemma_.lower()}.{obj[0].lemma_.lower()}")

    return list(set(elements))

def preprocess_specification(sentence):
    nouns, adjs, doc = linguistic_process(sentence)

    # UML element set = nouns + adjectives + mapped elements
    uml_element_set = list(set(nouns + adjs))

    return {
        "nouns": nouns,
        "adjectives": adjs,
        "uml_element_set": uml_element_set
    }

def parse_puml(puml_text):
    """
    Returns:
      classes: dict[className] -> list of raw attribute names (as in PUML, e.g. 'departTime')
      associations: dict[(clsA, clsB)] -> role_name_at_clsB_end (raw string)
    """
    classes = defaultdict(list)
    associations = {}

    lines = [ln.strip() for ln in puml_text.splitlines() if ln.strip() and not ln.strip().startswith("'")]
    current_class = None
    inside_class = False

    for line in lines:
        # start of class definition: `class Name{` or `class Name {`
        m = re.match(r'^class\s+(\w+)\s*\{', line)
        if m:
            current_class = m.group(1)
            inside_class = True
            continue

        # end of class
        if inside_class and line == "}":
            current_class = None
            inside_class = False
            continue

        # attribute/operation lines (inside class block)
        if inside_class and ":" in line:
            # pick the thing before the colon as attribute name (ignores type)
            attr = line.split(":", 1)[0].strip()
            classes[current_class].append(attr)
            continue

        # association lines: try to match patterns like:
        # A "roleA mul" -- "roleB mul" B
        m = re.search(r'(\w+)\s+"([^"]+)"\s*--\s*"([^"]+)"\s*(\w+)', line)
        if m:
            left_cls, left_role, right_role, right_cls = m.groups()
            # role text is like "origin 1" or "departingFlights *" -> take first token as role name
            left_role_name = left_role.split()[0] if left_role.strip() else ""
            right_role_name = right_role.split()[0] if right_role.strip() else ""
            # associations[(left,right)] should map to role name at RIGHT end (role on right class)
            associations[(left_cls, right_cls)] = right_role_name
            associations[(right_cls, left_cls)] = left_role_name
            continue

    return dict(classes), associations

def normalize_and_lemmatize_name(raw_name):
    """
    Normalize a property/role name:
    - split camelCase into words
    - remove non-alphanumerics
    - lemmatize tokens via spaCy
    - join lemmas into a single lowercase token (paper-style: 'departTime' -> 'departtime')
    """
    if not raw_name:
        return ""
    # split camelCase: "departTime" -> "depart Time"
    s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', raw_name)
    # replace non-alnum with space
    s = re.sub(r'[^0-9A-Za-z ]+', ' ', s)
    s = s.strip()
    if not s:
        return ""
    doc = nlp(s)
    lemmas = []
    for tok in doc:
        # keep alphabetic tokens (or numbers if needed)
        if tok.is_alpha or tok.like_num:
            lemmas.append(tok.lemma_.lower())
    # Join without spaces to mimic paper's formatting ('departtime')
    return "".join(lemmas)

def build_property_set_for_path(path, classes, associations):
    """
    path: list of class names in order e.g. ['Airline', 'Flight']
    classes: dict from parse_puml
    associations: dict from parse_puml
    Returns a set that contains:
      - class names (preserve original capitalization)
      - normalized & lemmatized attributes of ALL classes in the path
      - normalized & lemmatized role names for each transition in the path (role at the target class end)
    """
    prop_set = set()

    # 1) Add class names exactly as they appear
    for cls in path:
        prop_set.add(cls)

    # 2) Add attributes of ALL classes in the path (lemmatized/normalized)
    for cls in path:
        for raw_attr in classes.get(cls, []):
            name = raw_attr.split("(")[0].split(":")[0].strip()  # defensive: remove method params or types
            norm = normalize_and_lemmatize_name(name)
            if norm:
                prop_set.add(norm)

    # 3) Add role names for transitions: for src->dst use associations[(src,dst)] which stores role at DST end
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        role_raw = associations.get((src, dst))
        if role_raw:
            norm = normalize_and_lemmatize_name(role_raw)
            if norm:
                prop_set.add(norm)

    return prop_set

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets"""
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def cosine_sim_matrix(uml_element_set, uml_property_set):
    """
    Compute average cosine similarity between all pairs
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert to list for stable ordering
    elements = list(uml_element_set)
    properties = list(uml_property_set)

    # Get embeddings
    elem_emb = model.encode(elements, convert_to_tensor=True, normalize_embeddings=True)
    prop_emb = model.encode(properties, convert_to_tensor=True, normalize_embeddings=True)

    # Compute similarity matrix (elements x properties)
    sim_matrix = torch.matmul(elem_emb, prop_emb.T).cpu().numpy()

    # Average similarity
    return float(sim_matrix.mean())

def parse_puml_for_prompt(puml_text):
    """
    Parses PlantUML text into structured UML info:
    classes -> { name: { attributes: [], operations: [], associations: [] } }
    """
    classes = defaultdict(lambda: {"attributes": [], "operations": [], "associations": []})

    # --- Parse classes, attributes, and operations ---
    class_pattern = re.compile(r"class\s+(\w+)\s*\{([^}]*)\}", re.MULTILINE | re.DOTALL)
    attr_pattern = re.compile(r"(\w+)\s*:\s*(\w+)")
    op_pattern = re.compile(r"(\w+)\((.*?)\)")

    for class_match in class_pattern.finditer(puml_text):
        cname = class_match.group(1)
        body = class_match.group(2).strip()

        for attr_match in attr_pattern.finditer(body):
            attr, dtype = attr_match.groups()
            classes[cname]["attributes"].append((attr, dtype))

        for op_match in op_pattern.finditer(body):
            opname, params = op_match.groups()
            param_str = params.strip() if params else "void"
            classes[cname]["operations"].append((opname, param_str))

    # --- Parse associations ---
    assoc_pattern = re.compile(
        r'(\w+)\s+"?([\w\s\.]+)"?\s*--\s*"?([\w\s\.]+)"?\s*(\w+)'
    )
    for assoc_match in assoc_pattern.finditer(puml_text):
        src, src_role, tgt_role, tgt = assoc_match.groups()
        classes[src]["associations"].append((tgt, src_role.strip(), tgt_role.strip()))
        classes[tgt]["associations"].append((src, tgt_role.strip(), src_role.strip()))

    return classes

def build_prompt_for_path(spec_text, path, classes):
    """Builds a prompt using ALL unique classes in the simple path"""
    class_blocks = []
    seen = set()   # track already added classes
    
    for cls in path:
        if cls not in classes or cls in seen:
            continue  # skip if not found OR already processed
        seen.add(cls)

        info = classes[cls]

        # Attributes
        attr_lines = [f'        {{ "{attr}": "{dtype}" }}' for attr, dtype in info["attributes"]]
        attr_block = ",\n".join(attr_lines) if attr_lines else ""

        # Operations
        op_lines = [f'        {{ "{op}": "{params}" }}' for op, params in info["operations"]]
        op_block = ",\n".join(op_lines) if op_lines else ""

        # Associations
        assoc_lines = [
            f'        {{ "target": "{tgt}", "role": "{src_role}", "multiplicity": "{tgt_role}" }}'
            for tgt, src_role, tgt_role in info["associations"]
        ]
        assoc_block = ",\n".join(assoc_lines) if assoc_lines else ""

        class_block = f"""
-- UML properties of class {cls} 
{{ 
    "attributes": 
    [ 
{attr_block}
    ], 
    "operations": 
    [ 
{op_block}
    ], 
    "associations": 
    [ 
{assoc_block}
    ] 
}}"""
        class_blocks.append(class_block.strip())

    return f"""-- OCL specification 
{spec_text}

{chr(10).join(class_blocks)}

-- OCL constraint
"""

def load_puml_file(domain_name):
    """Load PlantUML file for the given domain"""
    puml_path = f"./pathocl_dataset/UML/{domain_name}/{domain_name}.puml"
    if not os.path.exists(puml_path):
        raise FileNotFoundError(f"PUML file not found: {puml_path}")
    
    with open(puml_path, 'r') as f:
        return f.read()

def load_specifications(spec_file_path, domain_name, spec_text=None):
    """Load specifications from JSON file"""
    if not os.path.exists(spec_file_path):
        raise FileNotFoundError(f"Specification file not found: {spec_file_path}")
    
    with open(spec_file_path, 'r') as f:
        specs_data = json.load(f)
    
    if domain_name not in specs_data:
        raise ValueError(f"Domain '{domain_name}' not found in specification file")
    
    domain_specs = specs_data[domain_name]["specifications"]
    
    if spec_text:
        if spec_text not in domain_specs:
            raise ValueError(f"Specification '{spec_text}' not found for domain '{domain_name}'")
        return {spec_text: domain_specs[spec_text]}
    else:
        return domain_specs

def generate_simple_paths(classes):
    """Generate simple paths from the class structure"""
    # This is a simplified path generation - you might want to implement more sophisticated path generation
    class_names = list(classes.keys())
    simple_paths = []
    
    # Generate all possible 2-class paths
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                simple_paths.append([class_names[i], class_names[j]])
    
    # Generate some 3-class paths (simplified)
    if len(class_names) >= 3:
        for i in range(min(3, len(class_names))):
            for j in range(min(3, len(class_names))):
                for k in range(min(3, len(class_names))):
                    if i != j and j != k and i != k:
                        simple_paths.append([class_names[i], class_names[j], class_names[k]])
    
    return simple_paths

def run_llama3_8b(user_prompt, hf_token):
    """Run Llama3-8B model"""
    system_prompt = "As a system designer with expertise in UML modeling and OCL constraints, your role is to assist the user in writing OCL constraints. The user will provide you with the following information: (1) The specification in natural language. (2) The UML classes and their properties (attributes, operations, associations). Your objective is to generate a valid OCL constraint according to the provided UML classes. Please do not provide explanation. Put your solution in a <OCL> tag."

    model_id = "meta-llama/Meta-Llama-3-8B"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build pipeline with the quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages = f"""
<|system|>
{system_prompt}
<|user|>
{user_prompt}
<|assistant|>
"""

    outputs = pipe(messages)
    return outputs[0]["generated_text"]

def run_qwen3_8b(user_prompt, hf_token):
    """Run Qwen3-8B model"""
    system_prompt = "As a system designer with expertise in UML modeling and OCL constraints, your role is to assist the user in writing OCL constraints. The user will provide you with the following information: (1) The specification in natural language. (2) The UML classes and their properties (attributes, operations, associations). Your objective is to generate a valid OCL constraint according to the provided UML classes. Please do not provide explanation. Put your solution in a <OCL> tag."

    model_id = "Qwen/Qwen3-8B"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build pipeline with the quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages_qwen = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{user_prompt}"}
    ]

    text = tokenizer.apply_chat_template(
        messages_qwen,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
         max_new_tokens=512
    )

    output = tokenizer.decode(
        generated_ids[0][len(model_inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    return output.strip()

def run_phi4_14b(user_prompt, hf_token):
    """Run Phi-4 model"""
    system_prompt = "As a system designer with expertise in UML modeling and OCL constraints, your role is to assist the user in writing OCL constraints. The user will provide you with the following information: (1) The specification in natural language. (2) The UML classes and their properties (attributes, operations, associations). Your objective is to generate a valid OCL constraint according to the provided UML classes. Please do not provide explanation. Put your solution in a <OCL> tag."

    model_id = "microsoft/phi-4"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build pipeline with the quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages_phi = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{user_prompt}"}
    ]

    outputs = pipe(messages_phi, max_new_tokens=512)
    return outputs[0]["generated_text"]

def run_gemma_7b(user_prompt, hf_token):
    """Run Gemma-7B model"""
    system_prompt = "As a system designer with expertise in UML modeling and OCL constraints, your role is to assist the user in writing OCL constraints. The user will provide you with the following information: (1) The specification in natural language. (2) The UML classes and their properties (attributes, operations, associations). Your objective is to generate a valid OCL constraint according to the provided UML classes. Please do not provide explanation. Put your solution in a <OCL> tag."

    model_id = "google/gemma-7b"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build pipeline with the quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages_gemma = f"""
<|system|>
{system_prompt}
<|user|>
{user_prompt}
"""

    outputs = pipe(messages_gemma, max_new_tokens=512)
    return outputs[0]["generated_text"]

def run_ministral_8b(user_prompt, hf_token):
    """Run Ministral-8B model"""
    system_prompt = "As a system designer with expertise in UML modeling and OCL constraints, your role is to assist the user in writing OCL constraints. The user will provide you with the following information: (1) The specification in natural language. (2) The UML classes and their properties (attributes, operations, associations). Your objective is to generate a valid OCL constraint according to the provided UML classes. Please do not provide explanation. Put your solution in a <OCL> tag."

    model_id = "mistralai/Ministral-8B-Instruct-2410"

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build pipeline with the quantized model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages_ministal = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{user_prompt}"}
    ]

    outputs = pipe(messages_ministal, max_new_tokens=512)
    return outputs[0]["generated_text"]

def main():
    parser = argparse.ArgumentParser(description="Generate OCL constraints from natural language specifications")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., Airport, Tournament)")
    parser.add_argument("--spec", required=True, help="Natural language specification text")
    parser.add_argument("--spec-file", required=True, help="Path to specifications JSON file")
    parser.add_argument("--llm", required=True, choices=["llama3-8b", "qwen3-8b", "phi4-14b", "gemma-7b", "ministral-8b"], 
                       help="LLM model to use")
    parser.add_argument("--hf-token", required=True, help="HuggingFace access token")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top paths to consider")
    
    args = parser.parse_args()
    
    # Login to HuggingFace
    login(args.hf_token)
    
    # Load data
    puml_text = load_puml_file(args.domain)
    specifications = load_specifications(args.spec_file, args.domain, args.spec)
    
    # Parse UML
    classes, associations = parse_puml(puml_text)
    
    # Generate simple paths - These can be derived from https://cs.gmu.edu:8443/offutt/coverage/GraphCoverage
    simple_paths = [
        ["Airport"],
        ["Flight"],
        ["Passenger"],
        ["Airline"],
        ["Airport","Flight"],
        ["Flight","Airport"],
        ["Flight","Passenger"],
        ["Passenger","Flight"],
        ["Airline","Flight"],
        ["Passenger","Airline"],
        ["Airline", "Passenger"],
        ["Airport","Flight","Airport"],
        ["Airport","Flight","Passenger"],
        ["Flight","Airport","Flight"],
        ["Flight","Passenger","Flight"],
        ["Flight","Passenger","Airline"],
        ["Passenger","Flight","Airport"],
        ["Passenger","Flight","Passenger"],
        ["Airline","Flight","Airport"],
        ["Airline","Flight","Passenger"],
        ["Passenger","Airline","Flight"],
        ["Airport","Flight","Passenger","Airline"],
        ["Flight","Passenger","Airline","Flight"],
        ["Airline","Flight","Passenger","Airline"],
        ["Passenger","Airline","Flight","Airport"],
        ["Passenger","Airline","Flight","Passenger"]
    ]
    
    # Preprocess specification
    spec_text = list(specifications.keys())[0]  # Get the first (and only) specification
    uml_element_set_data = preprocess_specification(spec_text)
    uml_element_set = set(uml_element_set_data['uml_element_set'])
    
    # Build UML property sets for all paths
    uml_property_sets = []
    for path in simple_paths:
        prop_set = build_property_set_for_path(path, classes, associations)
        uml_property_sets.append((path, prop_set))
    
    # Compute similarity scores
    scored_property_sets = []
    for i, (path, prop_set) in enumerate(uml_property_sets):
        jaccard = jaccard_similarity(uml_element_set, prop_set)
        cosine = cosine_sim_matrix(uml_element_set, prop_set)
        scored_property_sets.append((i, jaccard, cosine, path, prop_set))
    
    # Rank by descending similarity
    ranked = sorted(scored_property_sets, key=lambda x: x[1], reverse=True)
    
    # Parse UML for prompt building
    classes_prompt = parse_puml_for_prompt(puml_text)
    
    # Build prompts for top-k paths
    prompts = []
    for i in range(min(args.top_k, len(ranked))):
        idx, jscore, cscore, path, prop_set = ranked[i]
        prompt = build_prompt_for_path(spec_text, path, classes_prompt)
        prompts.append(prompt)
    
    user_prompt = prompts[0]  # Use the top-ranked path
    
    print("Generated Prompt:")
    print(user_prompt)
    print("\n" + "="*80 + "\n")
    
    # Run the selected LLM
    llm_functions = {
        "llama3-8b": run_llama3_8b,
        "qwen3-8b": run_qwen3_8b,
        "phi4-14b": run_phi4_14b,
        "gemma-7b": run_gemma_7b,
        "ministral-8b": run_ministral_8b
    }
    
    if args.llm in llm_functions:
        print(f"Running {args.llm}...")
        result = llm_functions[args.llm](user_prompt, args.hf_token)
        print("LLM Output:")
        print(result)
    else:
        print(f"Unknown LLM: {args.llm}")

if __name__ == "__main__":
    main()