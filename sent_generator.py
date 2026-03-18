
import re,random, nltk, string
# from collections import defaultdict, Counter
from nltk.corpus import brown
# from nltk import word_tokenize

# Download required NLTK data
nltk.download('brown')
nltk.download('punkt')

# Step 1 : Retrieve data & inputs
print("Enter the number of sentences (M):")
m = int(input())

while True:
    print("Enter your N-gram size (N) choose 2 or 3:")
    n = int(input())
    if n in [2, 3]:
        break
    print("Invalid input. Please try again.")

print(f"Success! You chose {n}.")

print("Enter your maximum sentence length (maxLen):")
maxlen = int(input())

if maxlen < n:
    raise ValueError("maxLen must be at least N to generate valid sentences.")

# brown.words() already returns word-level tokens.
tokens = brown.words()[:200000]


# Step 2 : Data Preprocessing
def preprocess_tokens(text):
    clean_tokens = []
    for word in text:
        # Convert to lowercase
        clean_word = word.lower()
        # Remove punctuation
        clean_word = re.sub(f"[{re.escape(string.punctuation)}]", "", clean_word)
        # Check if empty or if it is a stop word
        if len(clean_word) > 0 :
            clean_tokens.append(clean_word)
    vocabulary = set(clean_tokens)
    
    return clean_tokens,vocabulary

clean_tokens,vocab = preprocess_tokens(tokens)

print(f"Tokens after cleaning: {len(clean_tokens)}")
print(f"Unique vocabulary words: {len(vocab)}")


# Step 3 : Build the N-gram Models 
def build_ngram_models(tokens,n):
    ngram_counts={}
    prefix_counts={}
    for i in range(len(tokens) - n + 1): 
        pair = tuple(tokens[i:i+n])
        prefix = pair[:-1]
        
        if pair not in ngram_counts:
            ngram_counts[pair] = 1
        else:
            ngram_counts[pair] += 1
        
        if prefix not in prefix_counts:
            prefix_counts[prefix] = 1
        else:
            prefix_counts[prefix] += 1
    
    return prefix_counts,ngram_counts

prefix_counts,ngram_counts = build_ngram_models(clean_tokens,n)
print(f"Total unique {n}-grams: {len(ngram_counts)}")

# Step 4 : Develop the Sentence Generator
voc_sentences = []
while len(voc_sentences) < m:
    prefix = random.choice(list(prefix_counts.keys()))
    sent = list(prefix) 

    while len(sent) < maxlen:
        # Keep only observed continuations for the current prefix.
        candidates = [(k, v) for k, v in ngram_counts.items() if k[:-1] == prefix]
        if not candidates:
            break

        best_ngram = None
        highest_prob = -1.0
        for gram, count in candidates:
            prob = count / prefix_counts[prefix]
            if prob > highest_prob:
                highest_prob = prob
                best_ngram = gram

        if best_ngram is None:
            break

        sent.append(best_ngram[-1])
        # Update the prefix to be the last (n-1) words of the current sentence
        prefix = tuple(sent[-(n-1):])
        
    sentence_text = " ".join(sent)
    voc_sentences.append(sentence_text)
print("\n--- Generated Sentences ---\n")
print('\n'.join(voc_sentences))