import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Set, Tuple, Optional
import nltk
from collections import defaultdict, Counter
from functools import lru_cache
from dataclasses import dataclass, field
import os
from constants import CONTRACTIONS, REMOVE_PATTERNS, TIME_CONTEXTS, AMOUNT_CONTEXTS, REMOVED_WORDS # Import the constants

# Download NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

# Default configuration parameters (defined in the .py file)
DEFAULT_MIN_DF = 0.001
DEFAULT_IMPORTANT_TERMS = {
    'service', 'quality', 'customer', 'product', 'shipping',
    'recommend', 'happy', 'great', 'good', 'bad', 'terrible',
    'excellent', 'awful', 'amazing', 'horrible', 'best', 'worst',
    'love', 'hate', 'helpful', 'useless', 'complaint', 'thank'
}
DEFAULT_NGRAM_RANGE = (2, 5)
DEFAULT_TOP_N = 20

@dataclass
class TextProcessorConfig:
    """Configuration for text processing."""
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE
    top_n: int = DEFAULT_TOP_N
    min_df: float = DEFAULT_MIN_DF
    stop_words: List[str] = field(default_factory=lambda: [])
    important_terms: Set[str] = field(default_factory=lambda: DEFAULT_IMPORTANT_TERMS)

    def __post_init__(self):
        # Ensure important terms are not in stop words
        self.stop_words = [
            word for word in REMOVED_WORDS if word not in self.important_terms
        ]

def _process_chunk_helper(chunk: np.ndarray, processor: 'TextProcessor') -> pd.DataFrame:
    """Helper function to process a text chunk with a given config (for pickling)."""
    return processor.process_text_chunk(chunk)


class TextProcessor:
    def __init__(self, config: Optional[TextProcessorConfig] = None, num_processes: int = None, chunk_size: int = 10000, word_counts_file: str = 'word_counts.txt'):
        """Initialize TextProcessor with multiprocessing settings, configuration, and file path"""
        self.num_processes = num_processes or min(ProcessPoolExecutor()._max_workers, 8)
        self.chunk_size = chunk_size
        self.config = config or TextProcessorConfig()
        self.word_counts_file = word_counts_file

    def clean_text(self, text: str) -> str:
        """Aggressively clean text by removing special characters, normalizing, and applying stop word removal"""
        if not text or not isinstance(text, str):
            return ''

        text = text.lower().strip()

        # Handle contractions
        words = text.split()
        cleaned_words = [CONTRACTIONS.get(word, word) for word in words]
        text = ' '.join(cleaned_words).strip()

        # Apply all removal patterns
        for pattern, replacement in REMOVE_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Remove stop words, structural starts, and structural words, except for important terms
        words = text.split()
        cleaned_words = [
            word for word in words
            if word in self.config.important_terms or (
                word not in self.config.stop_words 
            )
        ]
        text = ' '.join(cleaned_words).strip()

        # Final step: Remove standalone 't' and backticks
        text = re.sub(r'\bt\b', '', text)
        text = re.sub(r'`', '', text)

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def process_single_text(self, text: str) -> Dict:
        """Process a single text entry with aggressive cleaning"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        original_words = text.split()
        processed_text = self.clean_text(text)
        processed_words = processed_text.split()

        replacements = []
        if len(processed_words) != len(original_words):
            replacements.append('word_count_changed')

        similarity = self.calculate_similarity_ratio(text, processed_text)
        nouns = self.get_nouns(processed_text)
        
        word_count_original = len(original_words)
        word_count_modified = len(processed_words)

        return {
            'original_text': text,
            'modified_text': processed_text,
            'replacements': replacements,
            'original_vs_modified_similarity': similarity,
            'nouns': nouns,
            'word_count_original': word_count_original,
            'word_count_modified': word_count_modified
        }
    
    def process_text_chunk(self, texts: np.ndarray) -> pd.DataFrame:
        """Process a chunk of texts"""
        results = []
        for text in texts:
             results.append(self.process_single_text(text))
            
        return pd.DataFrame(results)
    
    def calculate_similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts"""
        if not text1 or not text2:
            return 0.0

        # Clean both texts - Assuming you want to use a default config here
        text1 = self.clean_text(text1)
        text2 = self.clean_text(text2)

        if not text1 or not text2:
            return 0.0

        # Calculate word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        return intersection / union
    
    def generate_analysis_report(self, df: pd.DataFrame, text_column: str, processed_df: pd.DataFrame) -> None:
        """Generate analysis report with similarity scores and statistics"""

        comparison_df = processed_df.copy()
        comparison_df['words_removed'] = comparison_df['word_count_original'] - comparison_df['word_count_modified']
       
        print("\nSimilarity Analysis:")
        print(comparison_df[[text_column, 'original_vs_modified_similarity']])

        print("\nSummary of Modifications:")
        print(f"Average words removed: {comparison_df['words_removed'].mean():.2f}")
        print(f"Average similarity to original: {comparison_df['original_vs_modified_similarity'].mean():.2f}")
    
    @lru_cache(maxsize=10000)
    def get_nouns(self, text: str) -> Tuple[str, ...]:
        """Extract nouns from text with caching"""
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        return tuple(word for word, pos in tagged if pos.startswith('NN'))
    
    def analyze_nouns_by_rating(self, df: pd.DataFrame, text_column: str, rating_column: str, min_frequency: int = 5) -> Dict[int, List[Tuple[str, int]]]:
        """Analyze nouns grouped by rating"""
        grouped = df.groupby(rating_column)[text_column]
        noun_counts = defaultdict(Counter)

        for rating, texts in grouped:
             nouns = [self.get_nouns(str(text)) for text in texts]
             noun_counts[rating].update([noun for noun_list in nouns for noun in noun_list])

        return {
            rating: sorted(
                [(noun, count) for noun, count in counts.items() if count >= min_frequency],
                key=lambda x: x[1],
                reverse=True
            )
            for rating, counts in noun_counts.items()
        }
    
    def print_noun_analysis(self, noun_analysis: Dict[int, List[Tuple[str, int]]], top_n: int = 10) -> None:
        """Print noun analysis results"""
        for rating in sorted(noun_analysis.keys()):
            print(f"\nRating {rating} - Top {top_n} nouns:")
            print("\n".join(f"{noun}: {count}" for noun, count in noun_analysis[rating][:top_n]))
    
    def save_word_counts(self, word_counts: Counter):
        """Saves the given word counts to the specified file."""
        with open(self.word_counts_file, 'w', encoding='utf-8') as f:
             for word, count in word_counts.items():
                f.write(f"{word}\t{count}\n")

    def load_word_counts(self) -> Counter:
        """Loads word counts from the specified file, defaults to an empty counter if file is not found."""
        word_counts = Counter()
        if os.path.exists(self.word_counts_file):
           with open(self.word_counts_file, 'r', encoding='utf-8') as f:
                for line in f:
                   word, count = line.strip().split('\t')
                   word_counts[word] = int(count)
        return word_counts
    
    def parallel_frequency_analysis(self, processed_df: pd.DataFrame, text_column: str, exclude_words: Optional[Set[str]] = None) -> Dict:
        """
        Analyzes word frequencies in a text column of a DataFrame in parallel.
    
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        exclude_words: Optional[Set[str]]
            Set of words to exclude from the analysis.
        Returns:
        --------
        Dictionary containing analysis results:
            - total_unique_words (int): Number of unique words in dataset.
            - total_word_occurrences (int): Total count of all words.
            - top_10_words (List[Tuple[str, int]]): Top 10 most frequent words.
        """
        if exclude_words is None:
             exclude_words = set()
        
        word_counts = self.load_word_counts()
        texts = processed_df['modified_text'].values

        # Parallel processing of chunks
        num_chunks = min(self.num_processes, len(texts))
        if num_chunks > 0:
           chunks = np.array_split(texts, num_chunks)

           with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                results = executor.map(self._process_text_for_frequency, chunks, [exclude_words] * len(chunks))
                for chunk_result in results:
                    word_counts.update(chunk_result)
        
        self.save_word_counts(word_counts)
        total_unique_words = len(word_counts)
        total_word_occurrences = sum(word_counts.values())
        top_10_words = word_counts.most_common(10)

        return {
            'total_unique_words': total_unique_words,
            'total_word_occurrences': total_word_occurrences,
            'top_10_words': top_10_words,
            'word_counts': word_counts
        }
    
    def _process_text_for_frequency(self, texts: np.ndarray, exclude_words: Set[str]) -> Counter:
        """Helper function to process a chunk of texts for word frequency counting."""
        words = Counter()
        for text in texts:
             if not isinstance(text, str):
                 text = str(text)
             cleaned_text = self.clean_text(text)
             words.update([word for word in cleaned_text.split() if word not in exclude_words])
        return words
    
    def print_most_common_words(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        top_n: int = 50, 
        min_word_length: int = 3,
        exclude_words: Optional[Set[str]] = None
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Print and return the most common words across the entire dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        top_n : int, optional
            Number of top words to display
        min_word_length : int, optional
            Minimum length of words to consider
         exclude_words : Optional[Set[str]]
             Words to exclude from analysis.
        
        Returns:
        --------
        Dictionary of word frequencies
        """
        # Define excluded words set
        if exclude_words is None:
           exclude_words = {'at', 'in', 'an', 'on', 'the', 'a', 'from'}

        # Use the existing parallel frequency analysis
        frequency_results = self.parallel_frequency_analysis(df, text_column, exclude_words=exclude_words)
        
        # Filter and process word frequencies
        filtered_words = [
            (word, count) 
            for word, count in frequency_results['top_10_words']
            if len(word) >= min_word_length
        ]
        
        # Print results with formatting
        print("\n=== Most Common Words Across Dataset ===")
        print(f"Total Unique Words: {frequency_results['total_unique_words']}")
        print(f"Total Word Occurrences: {frequency_results['total_word_occurrences']}")
        print("\nTop {} Words:".format(top_n))
        
        for rank, (word, count) in enumerate(filtered_words[:top_n], 1):
            percentage = (count / frequency_results['total_word_occurrences']) * 100
            print(f"{rank}. {word}: {count} occurrences ({percentage:.2f}%)")
        
        return filtered_words
    
    def print_most_common_words_by_rating(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        rating_column: str, 
        top_n: int = 20, 
        min_word_length: int = 3,
        exclude_words: Optional[Set[str]] = None
    ) -> Dict[int, List[Tuple[str, int]]]:
        """
        Print and return the most common words for each rating level
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        text_column : str
            Name of the text column
        rating_column : str
            Name of the rating column
        top_n : int, optional
            Number of top words to display per rating
        min_word_length : int, optional
            Minimum length of words to consider
        exclude_words : Optional[Set[str]]
             Words to exclude from analysis.

        
        Returns:
        --------
        Dictionary of word frequencies by rating
        """
        # Validate column existence
        if text_column not in df.columns or rating_column not in df.columns:
            raise ValueError(f"Columns {text_column} or {rating_column} not found in DataFrame")
        
        # Define excluded words set
        if exclude_words is None:
           exclude_words = {'at', 'in', 'an', 'on', 'the', 'a', 'from'}
        # Group by rating
        rating_groups = df.groupby(rating_column)
        
        # Store results for each rating
        rating_word_frequencies = {}
        
        print(f"\n=== Most Common Words by {rating_column} ===")
        
        # Process each rating group
        for rating, group in rating_groups:
            # Process this group's texts
            frequency_results = self.parallel_frequency_analysis(group, text_column, exclude_words=exclude_words)
            
            # Filter words
            filtered_words = [
                (word, count) 
                for word, count in frequency_results['top_10_words']
                if len(word) >= min_word_length
            ]
            
            # Store results
            rating_word_frequencies[rating] = filtered_words
            
            # Print results
            print(f"\n--- Rating {rating} ---")
            print(f"Total Unique Words: {frequency_results['total_unique_words']}")
            print(f"Total Word Occurrences: {frequency_results['total_word_occurrences']}")
            print("\nTop {} Words:".format(top_n))
            
            for rank, (word, count) in enumerate(filtered_words[:top_n], 1):
                percentage = (count / frequency_results['total_word_occurrences']) * 100
                print(f"{rank}. {word}: {count} occurrences ({percentage:.2f}%)")
        
        return rating_word_frequencies

    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, batch_size: int = 1000) -> pd.DataFrame:
        """Process DataFrame in parallel using batched operations"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        results = []

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_rows)
                batch_texts = df[text_column].iloc[start_idx:end_idx].values

                num_chunks = min(self.num_processes, len(batch_texts))
                if num_chunks > 0:
                    chunks = np.array_split(batch_texts, num_chunks)
                    
                    # Use the helper function directly with executor.map
                    batch_results = list(executor.map(
                        _process_chunk_helper,
                        chunks,
                        [self] * len(chunks)
                    ))
                    results.extend(batch_results)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def process_text_for_comparison(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process the DataFrame for text comparison"""
        result_df = self.process_dataframe(df.copy(), text_column)
        return result_df

    def analyze_nouns(self, df: pd.DataFrame, text_column: str, rating_column: str, min_frequency: int = 5, top_n: int = 10) -> None:
        """Analyze nouns in text data with ratings"""
        results = self.analyze_nouns_by_rating(df, text_column, rating_column, min_frequency)
        self.print_noun_analysis(results, top_n)