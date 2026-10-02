import json
import faiss
import tomllib
from sentence_transformers import SentenceTransformer

def load_settings(config_path="example-settings.toml"):
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def main():
    settings = load_settings()
    embed_model_name = settings["cleaning"]["embed_model"]
    similarity_threshold = settings["cleaning"]["cosine_similarity_threshold"]
    
    print("Loading synthesized data from teammate...")
    with open("data_handoffs/synthesized_instructions.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print("Formatting text for embedding...")
    formatted_texts = []
    for item in dataset:
        text = f"{item.get('instruction', '')} {item.get('context', '')} {item.get('output', '')}"
        formatted_texts.append(text)
        
    print(f"Loading {embed_model_name} and encoding dataset...")
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(formatted_texts, convert_to_numpy=True, show_progress_bar=True)
    
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"Filtering diverse samples using threshold {similarity_threshold}...")
    distances, indices = index.search(embeddings, k=2)
    
    filtered_dataset = []
    removed_indices = set()
    
    for i in range(len(dataset)):
        if i in removed_indices:
            continue
            
        filtered_dataset.append(dataset[i])
        
        if distances[i][1] >= similarity_threshold:
            removed_indices.add(indices[i][1])
            
    print(f"Original size: {len(dataset)} | Filtered size: {len(filtered_dataset)}")
    
    with open("data_handoffs/final_filtered_dataset.json", "w", encoding="utf-8") as f:
        json.dump(filtered_dataset, f, ensure_ascii=False, indent=4)
    print("Filtered dataset ready for fine-tuning!")

if __name__ == "__main__":
    main()