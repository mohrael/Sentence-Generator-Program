# N-gram Sentence Generator

This project implements a **bigram and trigram sentence generator** using the **Brown Corpus** from NLTK.  
The generator uses a **greedy n-gram model** to create sentences based on the most probable next word, given the previous `(n-1)` words.

---

## 📚 Features

- Supports **bigram (n=2)** and **trigram (n=3)** models.
- Generates **M sentences** with a **maximum length of maxLen words**.
- Preprocesses text by:
  - Tokenizing sentences into words
  - Removing punctuation
  - Converting all words to lowercase
- Computes **n-gram counts** and **prefix counts** to calculate probabilities.
- Generates sentences **greedily based on highest probability**.

---

## 🧠 How It Works

1. **Data Preparation**
   - Load the first **200,000 tokens** from the Brown Corpus.
   - Preprocess tokens (lowercase, remove punctuation).

2. **N-gram Model Construction**
   - Build `ngram_counts` dictionary:
     ```
     ngram_counts[(w1, w2, ..., wn)] = count
     ```
   - Build `prefix_counts` dictionary:
     ```
     prefix_counts[(w1, w2, ..., w(n-1))] = count
     ```
   - These dictionaries are used to calculate probabilities:
     ```
     P(next_word | prefix) = ngram_counts[prefix + (next_word,)] / prefix_counts[prefix]
     ```

3. **Sentence Generation**
   - Pick a random **prefix** `(n-1 words)` from `prefix_counts`.
   - Iteratively select the **next word with highest probability**.
   - Append the next word to the sentence.
   - Slide the prefix forward (last n-1 words) and repeat until `maxLen` is reached.

---

## ⚙️ Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/n-gram-sentence-generator.git
cd n-gram-sentence-generator
