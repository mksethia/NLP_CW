#!/usr/bin/env python3
"""
Comprehensive NLP EDA Script for Don't Patronize Me Dataset
============================================================
This script performs extensive exploratory data analysis including:
1. Basic Statistical Profiling
2. Lexical Analysis
3. Semantic & Syntactic Exploration
4. Noise and Artifact Detection
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')

# For NLP processing
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab', quiet=True)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.util import ngrams

# Output file
OUTPUT_FILE = "eda_results.txt"

class EDALogger:
    """Logger to write results to both console and file"""
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w', encoding='utf-8')
        
    def log(self, message=""):
        print(message)
        self.file.write(str(message) + "\n")
        
    def close(self):
        self.file.close()

def load_data(logger):
    """Load the Don't Patronize Me dataset"""
    logger.log("=" * 80)
    logger.log("LOADING DATA")
    logger.log("=" * 80)
    
    # Load PCL dataset (skip first 4 rows which are disclaimer)
    pcl_columns = ['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label']
    df_pcl = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t', 
                          skiprows=4, names=pcl_columns, 
                          on_bad_lines='skip', engine='python')
    
    # Load categories dataset
    cat_columns = ['par_id', 'art_id', 'text', 'keyword', 'country_code', 
                   'span_start', 'span_finish', 'span_text', 'pcl_category', 'num_annotators']
    df_cat = pd.read_csv('dontpatronizeme_categories.tsv', sep='\t',
                          skiprows=4, names=cat_columns,
                          on_bad_lines='skip', engine='python')
    
    logger.log(f"PCL Dataset Shape: {df_pcl.shape}")
    logger.log(f"Categories Dataset Shape: {df_cat.shape}")
    logger.log("")
    
    return df_pcl, df_cat

def basic_statistical_profiling(df_pcl, logger):
    """Section 1: Basic Statistical Profiling"""
    logger.log("=" * 80)
    logger.log("1. BASIC STATISTICAL PROFILING")
    logger.log("=" * 80)
    
    # Clean text column
    texts = df_pcl['text'].dropna().astype(str)
    
    # Tokenize all texts
    logger.log("\n--- Token Count Analysis ---")
    token_counts = []
    char_counts = []
    sentence_counts = []
    
    for text in texts:
        tokens = word_tokenize(text.lower())
        token_counts.append(len(tokens))
        char_counts.append(len(text))
        sentences = sent_tokenize(text)
        sentence_counts.append(len(sentences))
    
    logger.log(f"Total number of documents: {len(texts)}")
    logger.log(f"\nToken (Word) Statistics:")
    logger.log(f"  Average tokens per document: {np.mean(token_counts):.2f}")
    logger.log(f"  Minimum tokens: {np.min(token_counts)}")
    logger.log(f"  Maximum tokens: {np.max(token_counts)}")
    logger.log(f"  Median tokens: {np.median(token_counts):.2f}")
    logger.log(f"  Std Dev: {np.std(token_counts):.2f}")
    
    logger.log(f"\nCharacter Statistics:")
    logger.log(f"  Average characters per document: {np.mean(char_counts):.2f}")
    logger.log(f"  Minimum characters: {np.min(char_counts)}")
    logger.log(f"  Maximum characters: {np.max(char_counts)}")
    
    logger.log(f"\nSentence Statistics:")
    logger.log(f"  Average sentences per document: {np.mean(sentence_counts):.2f}")
    logger.log(f"  Minimum sentences: {np.min(sentence_counts)}")
    logger.log(f"  Maximum sentences: {np.max(sentence_counts)}")
    
    # Vocabulary Size
    logger.log("\n--- Vocabulary Analysis ---")
    all_tokens = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        # Filter out punctuation-only tokens
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]
        all_tokens.extend(tokens)
    
    vocab = set(all_tokens)
    logger.log(f"Total tokens in corpus: {len(all_tokens):,}")
    logger.log(f"Unique vocabulary size: {len(vocab):,}")
    logger.log(f"Type-Token Ratio (TTR): {len(vocab)/len(all_tokens):.4f}")
    
    # Class Distribution
    logger.log("\n--- Class Distribution (Label Analysis) ---")
    label_counts = df_pcl['label'].value_counts().sort_index()
    logger.log(f"\nOriginal Labels (0-4 scale):")
    for label, count in label_counts.items():
        percentage = (count / len(df_pcl)) * 100
        logger.log(f"  Label {label}: {count:,} ({percentage:.2f}%)")
    
    # Binary classification grouping
    df_pcl['binary_label'] = df_pcl['label'].apply(lambda x: 'PCL' if x >= 2 else 'Non-PCL')
    binary_counts = df_pcl['binary_label'].value_counts()
    logger.log(f"\nBinary Classification (as per paper):")
    logger.log(f"  Non-PCL (labels 0,1): {binary_counts.get('Non-PCL', 0):,} ({binary_counts.get('Non-PCL', 0)/len(df_pcl)*100:.2f}%)")
    logger.log(f"  PCL (labels 2,3,4): {binary_counts.get('PCL', 0):,} ({binary_counts.get('PCL', 0)/len(df_pcl)*100:.2f}%)")
    
    # Class imbalance ratio
    if 'PCL' in binary_counts.index and 'Non-PCL' in binary_counts.index:
        imbalance_ratio = binary_counts['Non-PCL'] / binary_counts['PCL']
        logger.log(f"  Class Imbalance Ratio (Non-PCL:PCL): {imbalance_ratio:.2f}:1")
    
    # Keyword distribution
    logger.log("\n--- Keyword (Target Community) Distribution ---")
    keyword_counts = df_pcl['keyword'].value_counts()
    for keyword, count in keyword_counts.head(15).items():
        percentage = (count / len(df_pcl)) * 100
        logger.log(f"  {keyword}: {count:,} ({percentage:.2f}%)")
    
    # Country distribution
    logger.log("\n--- Country Code Distribution (Top 15) ---")
    country_counts = df_pcl['country_code'].value_counts()
    for country, count in country_counts.head(15).items():
        percentage = (count / len(df_pcl)) * 100
        logger.log(f"  {country}: {count:,} ({percentage:.2f}%)")
    
    return token_counts, all_tokens, vocab

def lexical_analysis(texts, all_tokens, vocab, logger):
    """Section 2: Lexical Analysis"""
    logger.log("\n" + "=" * 80)
    logger.log("2. LEXICAL ANALYSIS")
    logger.log("=" * 80)
    
    stop_words = set(stopwords.words('english'))
    
    # Word Frequency
    logger.log("\n--- Most Common Words (All) ---")
    word_freq = Counter(all_tokens)
    for word, count in word_freq.most_common(30):
        logger.log(f"  '{word}': {count:,}")
    
    # Non-stopword frequency
    logger.log("\n--- Most Common Words (Excluding Stopwords) ---")
    non_stop_tokens = [t for t in all_tokens if t.lower() not in stop_words]
    non_stop_freq = Counter(non_stop_tokens)
    for word, count in non_stop_freq.most_common(30):
        logger.log(f"  '{word}': {count:,}")
    
    # N-gram Analysis
    logger.log("\n--- Bigram Analysis (Most Common Word Pairs) ---")
    bigrams_list = list(ngrams(all_tokens, 2))
    bigram_freq = Counter(bigrams_list)
    for bigram, count in bigram_freq.most_common(25):
        logger.log(f"  {bigram}: {count:,}")
    
    logger.log("\n--- Trigram Analysis (Most Common Word Triplets) ---")
    trigrams_list = list(ngrams(all_tokens, 3))
    trigram_freq = Counter(trigrams_list)
    for trigram, count in trigram_freq.most_common(25):
        logger.log(f"  {trigram}: {count:,}")
    
    # Stop Word Density
    logger.log("\n--- Stop Word Analysis ---")
    stop_word_count = sum(1 for t in all_tokens if t.lower() in stop_words)
    stop_word_density = stop_word_count / len(all_tokens) * 100
    logger.log(f"Total stop words: {stop_word_count:,}")
    logger.log(f"Stop word density: {stop_word_density:.2f}%")
    logger.log(f"Content word density: {100 - stop_word_density:.2f}%")
    
    # Most common stop words
    logger.log("\n--- Most Common Stop Words ---")
    stop_tokens = [t for t in all_tokens if t.lower() in stop_words]
    stop_freq = Counter(stop_tokens)
    for word, count in stop_freq.most_common(15):
        logger.log(f"  '{word}': {count:,}")
    
    # Word length distribution
    logger.log("\n--- Word Length Distribution ---")
    word_lengths = [len(w) for w in all_tokens]
    logger.log(f"Average word length: {np.mean(word_lengths):.2f} characters")
    logger.log(f"Median word length: {np.median(word_lengths):.2f} characters")
    length_dist = Counter(word_lengths)
    logger.log("Word length frequency:")
    for length in sorted(length_dist.keys())[:15]:
        count = length_dist[length]
        logger.log(f"  {length} chars: {count:,} words")
    
    return word_freq, non_stop_freq

def semantic_syntactic_exploration(texts, logger, sample_size=1000):
    """Section 3: Semantic & Syntactic Exploration"""
    logger.log("\n" + "=" * 80)
    logger.log("3. SEMANTIC & SYNTACTIC EXPLORATION")
    logger.log("=" * 80)
    
    # Sample texts for intensive analysis (NER/POS are computationally expensive)
    sample_texts = texts.sample(min(sample_size, len(texts)), random_state=42).tolist()
    
    # POS Tagging Analysis
    logger.log(f"\n--- Part-of-Speech (POS) Analysis (sample of {len(sample_texts)} docs) ---")
    all_pos_tags = []
    
    for text in sample_texts:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        all_pos_tags.extend([tag for word, tag in pos_tags])
    
    pos_freq = Counter(all_pos_tags)
    logger.log("\nPOS Tag Distribution (Top 20):")
    
    # POS tag explanations
    pos_explanations = {
        'NN': 'Noun (singular)',
        'NNS': 'Noun (plural)',
        'NNP': 'Proper Noun (singular)',
        'NNPS': 'Proper Noun (plural)',
        'VB': 'Verb (base form)',
        'VBD': 'Verb (past tense)',
        'VBG': 'Verb (gerund)',
        'VBN': 'Verb (past participle)',
        'VBP': 'Verb (non-3rd person)',
        'VBZ': 'Verb (3rd person)',
        'JJ': 'Adjective',
        'JJR': 'Adjective (comparative)',
        'JJS': 'Adjective (superlative)',
        'RB': 'Adverb',
        'RBR': 'Adverb (comparative)',
        'RBS': 'Adverb (superlative)',
        'DT': 'Determiner',
        'IN': 'Preposition',
        'CC': 'Coordinating Conjunction',
        'PRP': 'Personal Pronoun',
        'PRP$': 'Possessive Pronoun',
        'MD': 'Modal',
        'TO': 'to',
        'CD': 'Cardinal Number',
        'WDT': 'Wh-determiner',
        'WP': 'Wh-pronoun',
        'WRB': 'Wh-adverb',
    }
    
    for tag, count in pos_freq.most_common(20):
        explanation = pos_explanations.get(tag, 'Other')
        percentage = count / len(all_pos_tags) * 100
        logger.log(f"  {tag} ({explanation}): {count:,} ({percentage:.2f}%)")
    
    # Noun vs Verb ratio
    noun_count = sum(c for t, c in pos_freq.items() if t.startswith('NN'))
    verb_count = sum(c for t, c in pos_freq.items() if t.startswith('VB'))
    adj_count = sum(c for t, c in pos_freq.items() if t.startswith('JJ'))
    
    logger.log(f"\nAggregate POS Statistics:")
    logger.log(f"  Total Nouns: {noun_count:,} ({noun_count/len(all_pos_tags)*100:.2f}%)")
    logger.log(f"  Total Verbs: {verb_count:,} ({verb_count/len(all_pos_tags)*100:.2f}%)")
    logger.log(f"  Total Adjectives: {adj_count:,} ({adj_count/len(all_pos_tags)*100:.2f}%)")
    if verb_count > 0:
        logger.log(f"  Noun-to-Verb Ratio: {noun_count/verb_count:.2f}:1")
    
    # Named Entity Recognition
    logger.log(f"\n--- Named Entity Recognition (NER) Analysis ---")
    ner_counts = Counter()
    entity_examples = {entity_type: [] for entity_type in 
                       ['PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT']}
    
    for text in sample_texts[:500]:  # Smaller sample for NER (very slow)
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_text = ' '.join(c[0] for c in chunk)
                ner_counts[entity_type] += 1
                if entity_type in entity_examples and len(entity_examples[entity_type]) < 5:
                    entity_examples[entity_type].append(entity_text)
    
    logger.log("\nNamed Entity Distribution:")
    for entity_type, count in ner_counts.most_common():
        examples = entity_examples.get(entity_type, [])
        example_str = f" (e.g., {', '.join(examples[:3])})" if examples else ""
        logger.log(f"  {entity_type}: {count:,}{example_str}")
    
    return pos_freq, ner_counts

def noise_artifact_detection(df_pcl, texts, token_counts, logger):
    """Section 4: Identifying Noise and Artifacts"""
    logger.log("\n" + "=" * 80)
    logger.log("4. NOISE AND ARTIFACT DETECTION")
    logger.log("=" * 80)
    
    # Duplicate Detection
    logger.log("\n--- Duplicate Analysis ---")
    total_rows = len(df_pcl)
    
    # Exact duplicates
    exact_duplicates = df_pcl[df_pcl.duplicated(subset=['text'], keep=False)]
    exact_dup_count = df_pcl.duplicated(subset=['text']).sum()
    logger.log(f"Exact text duplicates: {exact_dup_count:,} ({exact_dup_count/total_rows*100:.2f}%)")
    
    # Near-duplicates (same par_id)
    par_id_duplicates = df_pcl.duplicated(subset=['par_id']).sum()
    logger.log(f"Duplicate par_ids: {par_id_duplicates:,}")
    
    # Show some duplicate examples
    if exact_dup_count > 0:
        logger.log("\nSample duplicate texts (first 3 groups):")
        dup_texts = df_pcl[df_pcl.duplicated(subset=['text'], keep=False)]['text'].value_counts().head(3)
        for i, (text, count) in enumerate(dup_texts.items(), 1):
            logger.log(f"  Group {i} (appears {count} times): '{text[:100]}...'")
    
    # Special Characters and HTML Detection
    logger.log("\n--- Special Characters & HTML Artifacts ---")
    
    patterns = {
        'HTML entities': r'&[a-zA-Z]+;|&#\d+;',
        'HTML tags': r'<[^>]+>',
        'URLs': r'https?://\S+|www\.\S+',
        'Email addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'Newlines/tabs': r'\\n|\\t|\\r',
        'Multiple spaces': r'\s{3,}',
        'Special quotes': r'[""''„‚«»]',
        'Non-ASCII chars': r'[^\x00-\x7F]+',
        'Hashtags': r'#\w+',
        'Mentions': r'@\w+',
        'Numbers only': r'^\d+$',
    }
    
    artifact_counts = {}
    artifact_examples = {}
    
    for pattern_name, pattern in patterns.items():
        matches = texts.str.contains(pattern, regex=True, na=False)
        count = matches.sum()
        artifact_counts[pattern_name] = count
        
        # Get examples
        if count > 0:
            examples = texts[matches].head(2).tolist()
            artifact_examples[pattern_name] = examples
    
    for pattern_name, count in sorted(artifact_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(texts) * 100
        logger.log(f"  {pattern_name}: {count:,} documents ({percentage:.2f}%)")
        if pattern_name in artifact_examples and artifact_examples[pattern_name]:
            # Find the actual match in the text
            example = artifact_examples[pattern_name][0][:150]
            logger.log(f"    Example: '{example}...'")
    
    # Specific problematic patterns
    logger.log("\n--- Specific Pattern Detection ---")
    
    # Question marks (interrogative sentences)
    question_count = texts.str.contains(r'\?', na=False).sum()
    logger.log(f"Documents with questions: {question_count:,} ({question_count/len(texts)*100:.2f}%)")
    
    # Exclamation marks (emphatic)
    exclaim_count = texts.str.contains(r'!', na=False).sum()
    logger.log(f"Documents with exclamations: {exclaim_count:,} ({exclaim_count/len(texts)*100:.2f}%)")
    
    # Quoted text
    quote_count = texts.str.contains(r'["\'].*["\']', na=False).sum()
    logger.log(f"Documents with quoted text: {quote_count:,} ({quote_count/len(texts)*100:.2f}%)")
    
    # Outlier Detection
    logger.log("\n--- Outlier Analysis (Sequence Length) ---")
    
    token_array = np.array(token_counts)
    q1 = np.percentile(token_array, 25)
    q3 = np.percentile(token_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_low = (token_array < lower_bound).sum()
    outliers_high = (token_array > upper_bound).sum()
    
    logger.log(f"Q1 (25th percentile): {q1:.2f} tokens")
    logger.log(f"Q3 (75th percentile): {q3:.2f} tokens")
    logger.log(f"IQR: {iqr:.2f}")
    logger.log(f"Lower bound (Q1 - 1.5*IQR): {max(0, lower_bound):.2f}")
    logger.log(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.2f}")
    logger.log(f"Outliers (too short): {outliers_low:,} ({outliers_low/len(token_counts)*100:.2f}%)")
    logger.log(f"Outliers (too long): {outliers_high:,} ({outliers_high/len(token_counts)*100:.2f}%)")
    
    # Show extreme examples
    logger.log("\n--- Extreme Length Examples ---")
    
    # Create a dataframe with texts and their token counts
    texts_list = texts.tolist()
    df_with_tokens = pd.DataFrame({
        'text': texts_list,
        'token_count': token_counts[:len(texts_list)]
    })
    
    logger.log("\nShortest documents:")
    shortest = df_with_tokens.nsmallest(5, 'token_count')
    for idx, row in shortest.iterrows():
        text_preview = str(row['text'])[:100]
        logger.log(f"  [{row['token_count']} tokens] '{text_preview}...'")
    
    logger.log("\nLongest documents:")
    longest = df_with_tokens.nlargest(5, 'token_count')
    for idx, row in longest.iterrows():
        text_preview = str(row['text'])[:100]
        logger.log(f"  [{row['token_count']} tokens] '{text_preview}...'")
    
    # Empty or near-empty texts
    empty_count = texts.str.strip().eq('').sum()
    very_short = (df_with_tokens['token_count'] <= 3).sum()
    logger.log(f"\nEmpty texts: {empty_count:,}")
    logger.log(f"Very short texts (≤3 tokens): {very_short:,}")

def analyze_categories(df_cat, logger):
    """Analyze the PCL categories dataset"""
    logger.log("\n" + "=" * 80)
    logger.log("5. PCL CATEGORIES ANALYSIS")
    logger.log("=" * 80)
    
    logger.log(f"\nTotal category annotations: {len(df_cat):,}")
    
    # Category distribution
    logger.log("\n--- PCL Category Distribution ---")
    cat_counts = df_cat['pcl_category'].value_counts()
    for cat, count in cat_counts.items():
        percentage = count / len(df_cat) * 100
        logger.log(f"  {cat}: {count:,} ({percentage:.2f}%)")
    
    # Annotator agreement
    logger.log("\n--- Annotator Agreement Distribution ---")
    annotator_counts = df_cat['num_annotators'].value_counts().sort_index()
    for num, count in annotator_counts.items():
        percentage = count / len(df_cat) * 100
        logger.log(f"  {num} annotator(s): {count:,} ({percentage:.2f}%)")
    
    # Span length analysis
    logger.log("\n--- PCL Span Length Analysis ---")
    df_cat['span_length'] = df_cat['span_finish'] - df_cat['span_start']
    span_lengths = df_cat['span_length'].dropna()
    logger.log(f"Average span length: {span_lengths.mean():.2f} characters")
    logger.log(f"Median span length: {span_lengths.median():.2f} characters")
    logger.log(f"Min span length: {span_lengths.min():.2f}")
    logger.log(f"Max span length: {span_lengths.max():.2f}")
    
    # Sample PCL spans by category
    logger.log("\n--- Sample PCL Spans by Category ---")
    for cat in cat_counts.index[:5]:  # Top 5 categories
        logger.log(f"\n{cat}:")
        samples = df_cat[df_cat['pcl_category'] == cat]['span_text'].dropna().head(3)
        for sample in samples:
            logger.log(f"  - '{sample[:80]}...'") if len(str(sample)) > 80 else logger.log(f"  - '{sample}'")

def generate_recommendations(logger, binary_counts, vocab_size, stop_density):
    """Generate modeling recommendations based on EDA findings"""
    logger.log("\n" + "=" * 80)
    logger.log("6. MODELING RECOMMENDATIONS")
    logger.log("=" * 80)
    
    logger.log("\n--- Based on EDA Findings ---")
    
    # Class imbalance
    if 'PCL' in binary_counts.index and 'Non-PCL' in binary_counts.index:
        imbalance_ratio = binary_counts['Non-PCL'] / binary_counts['PCL']
        if imbalance_ratio > 3:
            logger.log(f"\n⚠️  CLASS IMBALANCE DETECTED (Ratio: {imbalance_ratio:.1f}:1)")
            logger.log("   Recommendations:")
            logger.log("   - Use class weights in loss function")
            logger.log("   - Consider oversampling (SMOTE) or undersampling")
            logger.log("   - Use F1-score, not accuracy, as primary metric")
            logger.log("   - Consider focal loss for deep learning models")
    
    # Vocabulary
    logger.log(f"\n📊 VOCABULARY SIZE: {vocab_size:,}")
    logger.log("   Recommendations:")
    if vocab_size > 50000:
        logger.log("   - Large vocabulary: consider subword tokenization (BPE, WordPiece)")
    logger.log("   - Embedding dimension: 100-300 for this vocabulary size")
    logger.log("   - Consider pre-trained embeddings (GloVe, FastText) for better generalization")
    
    # Stop words
    logger.log(f"\n🔤 STOP WORD DENSITY: {stop_density:.1f}%")
    if stop_density > 45:
        logger.log("   - High stop word density: consider removing stop words for traditional ML")
        logger.log("   - For transformers: keep stop words (they capture important context)")
    
    logger.log("\n📝 GENERAL RECOMMENDATIONS:")
    logger.log("   - Max sequence length: Consider 95th percentile of token distribution")
    logger.log("   - For PCL detection: Context is crucial, use longer sequences")
    logger.log("   - Consider fine-tuning BERT/RoBERTa for this task")
    logger.log("   - Use stratified train/test split due to class imbalance")
    logger.log("   - Remove exact duplicates before splitting to avoid data leakage")

def main():
    """Main function to run all EDA analyses"""
    logger = EDALogger(OUTPUT_FILE)
    
    logger.log("=" * 80)
    logger.log("NLP EXPLORATORY DATA ANALYSIS REPORT")
    logger.log("Don't Patronize Me! Dataset")
    logger.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log("=" * 80)
    
    try:
        # Load data
        df_pcl, df_cat = load_data(logger)
        texts = df_pcl['text'].dropna().astype(str)
        
        # 1. Basic Statistical Profiling
        token_counts, all_tokens, vocab = basic_statistical_profiling(df_pcl, logger)
        
        # 2. Lexical Analysis
        word_freq, non_stop_freq = lexical_analysis(texts, all_tokens, vocab, logger)
        
        # 3. Semantic & Syntactic Exploration
        pos_freq, ner_counts = semantic_syntactic_exploration(texts, logger)
        
        # 4. Noise and Artifact Detection
        noise_artifact_detection(df_pcl, texts, token_counts, logger)
        
        # 5. Categories Analysis
        analyze_categories(df_cat, logger)
        
        # 6. Recommendations
        stop_words = set(stopwords.words('english'))
        stop_word_count = sum(1 for t in all_tokens if t.lower() in stop_words)
        stop_density = stop_word_count / len(all_tokens) * 100
        
        binary_counts = df_pcl['label'].apply(lambda x: 'PCL' if x >= 2 else 'Non-PCL').value_counts()
        generate_recommendations(logger, binary_counts, len(vocab), stop_density)
        
        logger.log("\n" + "=" * 80)
        logger.log("EDA COMPLETE")
        logger.log(f"Results saved to: {OUTPUT_FILE}")
        logger.log("=" * 80)
        
    except Exception as e:
        logger.log(f"\nERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
    
    finally:
        logger.close()

if __name__ == "__main__":
    main()
