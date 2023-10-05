import pickle

def build_vocab(data):
    # Flattening the list of lists
    all_texts = [item for sublist in data['extracted_texts'] for item in sublist]
    
    # Extracting tokens
    all_tokens = [token for text in all_texts for token in text.split() if len(text.split()) > 0]
    
    # Building vocabulary
    unique_tokens = set(all_tokens)
    word_to_id = {token: i for i, token in enumerate(unique_tokens)}
    
    # Ensure the <UNK> token is in the dictionary
    word_to_id["<UNK>"] = len(word_to_id)
    
    return word_to_id


# Load the data
with open("D:\\new-data-real.pickle", 'rb') as f:
    data = pickle.load(f)

word_to_id = build_vocab(data)
vocab_size = len(word_to_id)
print(f"The vocabulary size is: {vocab_size}")
# Save the vocabulary
with open("D:\\new-vocab-real.pickle", 'wb') as f:
    pickle.dump(word_to_id, f)

print("Vocabulary saved to vocab.pkl")