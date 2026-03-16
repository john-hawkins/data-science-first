"""
Text Quality Metrics Library
A collection of functions for measuring readability, vocabulary level, and logical integrity of text.
"""

import re
import math
from collections import Counter
from typing import Dict, Any


# =============================================================================
# READABILITY METRICS
# =============================================================================

def flesch_reading_ease(text: str, features: Dict[str, Any] = None) -> float:
    """
    Flesch Reading Ease Score
    
    Measures text readability on a scale of 0-100 (higher = more readable).
    Score interpretation:
    - 90-100: Very easy to read
    - 80-90: Easy to read  
    - 70-80: Fairly easy to read
    - 60-70: Standard
    - 50-60: Fairly difficult to read
    - 30-50: Difficult to read
    - 0-30: Very difficult to read
    
    Reference: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    syllables = sum(_count_syllables(word) for word in words)
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return max(0.0, min(100.0, score))


def flesch_kincaid_grade_level(text: str, features: Dict[str, Any] = None) -> float:
    """
    Flesch-Kincaid Grade Level
    
    Estimates the U.S. school grade level needed to understand the text.
    
    Reference: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    syllables = sum(_count_syllables(word) for word in words)
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    return max(0.0, grade_level)


def automated_readability_index(text: str, features: Dict[str, Any] = None) -> float:
    """
    Automated Readability Index (ARI)
    
    Estimates the grade level needed to comprehend the text based on 
    characters per word and words per sentence.
    
    Reference: https://en.wikipedia.org/wiki/Automated_readability_index
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text)
    characters = len(re.sub(r'\s', '', text))
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    ari = 4.71 * (characters / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43
    return max(0.0, ari)


def gunning_fog_index(text: str, features: Dict[str, Any] = None) -> float:
    """
    Gunning Fog Index
    
    Estimates the years of formal education needed to understand the text.
    Complex words are defined as words with 3 or more syllables.
    
    Reference: https://en.wikipedia.org/wiki/Gunning_fog_index
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    complex_words = [word for word in words if _count_syllables(word) >= 3]
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    percent_complex = (len(complex_words) / len(words)) * 100
    
    fog_index = 0.4 * (avg_sentence_length + percent_complex)
    return fog_index


def smog_readability(text: str, features: Dict[str, Any] = None) -> float:
    """
    SMOG Readability Formula
    
    Simple Measure of Gobbledygook - estimates years of education needed.
    Traditionally requires 30+ sentences for accuracy.
    
    Reference: https://en.wikipedia.org/wiki/SMOG
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    complex_words = [word for word in words if _count_syllables(word) >= 3]
    
    if len(sentences) < 3:
        return 0.0
    
    # Standard SMOG formula
    polysyllable_count = len(complex_words)
    smog_score = 1.043 * math.sqrt(polysyllable_count * (30 / len(sentences))) + 3.1291
    
    return max(0.0, smog_score)


# =============================================================================
# VOCABULARY LEVEL METRICS
# =============================================================================

def type_token_ratio(text: str, features: Dict[str, Any] = None) -> float:
    """
    Type-Token Ratio (TTR)
    
    Measures lexical diversity as the ratio of unique words to total words.
    Higher values indicate greater vocabulary diversity.
    
    Reference: https://en.wikipedia.org/wiki/Lexical_diversity
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return 0.0
    
    unique_words = set(words)
    return len(unique_words) / len(words)


def moving_average_ttr(text: str, features: Dict[str, Any] = None, window_size: int = 100) -> float:
    """
    Moving Average Type-Token Ratio (MATTR)
    
    Calculates TTR over moving windows to reduce text length dependency.
    More stable measure of lexical diversity than simple TTR.
    
    Reference: Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: 
    The moving-average type–token ratio (MATTR). Journal of quantitative linguistics, 17(2), 94-100.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < window_size:
        return type_token_ratio(text, features)
    
    ttrs = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        unique_words = set(window)
        ttrs.append(len(unique_words) / window_size)
    
    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def yule_k_characteristic(text: str, features: Dict[str, Any] = None) -> float:
    """
    Yule's K Characteristic
    
    Measures vocabulary richness independent of text length.
    Lower values indicate higher vocabulary diversity.
    
    Reference: https://en.wikipedia.org/wiki/Yule%27s_K
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return 0.0
    
    word_freq = Counter(words)
    freq_freq = Counter(word_freq.values())
    
    n = len(words)
    sum_freq_squared = sum(freq * (count ** 2) for count, freq in freq_freq.items())
    
    if n <= 1:
        return 0.0
    
    k = 10000 * (sum_freq_squared - n) / (n * n)
    return k


def average_word_length(text: str, features: Dict[str, Any] = None) -> float:
    """
    Average Word Length
    
    Mean number of characters per word, indicating vocabulary complexity.
    Longer words generally indicate more sophisticated vocabulary.
    
    Reference: Standard linguistic measure used in readability research.
    """
    words = re.findall(r'\b\w+\b', text)
    
    if len(words) == 0:
        return 0.0
    
    total_chars = sum(len(word) for word in words)
    return total_chars / len(words)


def syllable_complexity(text: str, features: Dict[str, Any] = None) -> float:
    """
    Average Syllables Per Word
    
    Measures phonological complexity of vocabulary.
    Higher values indicate more complex vocabulary.
    
    Reference: Used in various readability formulas as vocabulary complexity indicator.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return 0.0
    
    total_syllables = sum(_count_syllables(word) for word in words)
    return total_syllables / len(words)


# =============================================================================
# LOGICAL INTEGRITY METRICS
# =============================================================================

def logical_connector_density(text: str, features: Dict[str, Any] = None) -> float:
    """
    Logical Connector Density
    
    Measures the frequency of logical connectors (therefore, however, because, etc.)
    which indicate structured argumentation and logical flow.
    
    Reference: Crossley, S. A., & McNamara, D. S. (2014). Does writing development 
    equal writing quality? A computational investigation of syntactic complexity.
    """
    logical_connectors = {
        'therefore', 'thus', 'hence', 'consequently', 'accordingly', 'so',
        'however', 'nevertheless', 'nonetheless', 'although', 'though', 'while',
        'because', 'since', 'as', 'for', 'due to', 'owing to',
        'furthermore', 'moreover', 'additionally', 'also', 'besides',
        'in contrast', 'on the other hand', 'conversely', 'alternatively',
        'similarly', 'likewise', 'in the same way', 'equally',
        'first', 'second', 'third', 'finally', 'in conclusion', 'to summarize'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    text_lower = text.lower()
    
    if len(words) == 0:
        return 0.0
    
    connector_count = 0
    for connector in logical_connectors:
        if ' ' in connector:
            connector_count += len(re.findall(r'\b' + re.escape(connector) + r'\b', text_lower))
        else:
            connector_count += words.count(connector)
    
    return connector_count / len(words) * 1000  # Per 1000 words


def argument_structure_score(text: str, features: Dict[str, Any] = None) -> float:
    """
    Argument Structure Score
    
    Evaluates the presence of argumentative discourse markers that indicate
    claims, evidence, and reasoning structure.
    
    Reference: Based on argument mining research, e.g., Stab, C., & Gurevych, I. (2017). 
    Parsing argumentation structures in persuasive essays. Computational Linguistics, 43(3), 619-659.
    """
    claim_markers = {
        'i argue', 'i claim', 'i believe', 'i contend', 'my position', 'i propose',
        'it is clear', 'obviously', 'certainly', 'undoubtedly', 'surely'
    }
    
    evidence_markers = {
        'according to', 'research shows', 'studies indicate', 'data suggests',
        'evidence shows', 'statistics show', 'for example', 'for instance',
        'specifically', 'in particular', 'research by', 'study by'
    }
    
    reasoning_markers = {
        'this shows', 'this indicates', 'this suggests', 'this proves',
        'this demonstrates', 'we can conclude', 'it follows', 'this means'
    }
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if len(words) == 0:
        return 0.0
    
    total_markers = 0
    for markers in [claim_markers, evidence_markers, reasoning_markers]:
        for marker in markers:
            total_markers += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
    
    return total_markers / len(words) * 1000  # Per 1000 words


def coherence_score(text: str, features: Dict[str, Any] = None) -> float:
    """
    Sentence Coherence Score
    
    Measures textual coherence based on sentence length variation and
    discourse markers that indicate logical flow between sentences.
    
    Reference: Based on coherence measures in McNamara, D. S., et al. (2014). 
    Automated evaluation of text and discourse with Coh-Metrix. Cambridge University Press.
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0
    
    # Measure sentence length variation (lower variation = better coherence)
    sentence_lengths = [len(re.findall(r'\b\w+\b', sent)) for sent in sentences]
    
    if len(sentence_lengths) == 0:
        return 0.0
    
    mean_length = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((length - mean_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
    
    # Coefficient of variation (normalized)
    if mean_length == 0:
        length_consistency = 0
    else:
        length_consistency = 1 - min(1.0, math.sqrt(variance) / mean_length)
    
    # Count transition markers between sentences
    transition_markers = {
        'first', 'second', 'third', 'next', 'then', 'finally', 'subsequently',
        'previously', 'earlier', 'later', 'meanwhile', 'simultaneously',
        'in addition', 'furthermore', 'moreover', 'however', 'nevertheless',
        'on the contrary', 'in contrast', 'similarly', 'likewise'
    }
    
    transition_count = 0
    text_lower = text.lower()
    for marker in transition_markers:
        transition_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
    
    transition_density = transition_count / len(sentences) if len(sentences) > 0 else 0
    
    # Combine metrics (weighted average)
    coherence = 0.7 * length_consistency + 0.3 * min(1.0, transition_density)
    return coherence


def contradiction_detection_score(text: str, features: Dict[str, Any] = None) -> float:
    """
    Contradiction Detection Score
    
    Identifies potential contradictions by looking for negation patterns
    and conflicting statements. Lower scores indicate fewer contradictions.
    
    Reference: Simplified approach based on contradiction detection research,
    e.g., de Marneffe, M. C., et al. (2008). Finding contradictions in text.
    """
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip().lower() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 1.0  # No contradictions possible with fewer than 2 sentences
    
    negation_patterns = [
        r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bneither\b', r'\bnor\b',
        r'\bnothing\b', r'\bnobody\b', r'\bnowhere\b', r'\bhardly\b',
        r'\bscarcely\b', r'\bbarely\b', r"n't\b"
    ]
    
    contradiction_indicators = [
        r'\bbut\b', r'\bhowever\b', r'\balthough\b', r'\bthough\b',
        r'\bnevertheless\b', r'\bnonetheless\b', r'\bconversely\b',
        r'\bon the contrary\b', r'\bin contrast\b', r'\byet\b'
    ]
    
    negation_count = 0
    contradiction_count = 0
    
    for sentence in sentences:
        for pattern in negation_patterns:
            negation_count += len(re.findall(pattern, sentence))
        
        for pattern in contradiction_indicators:
            contradiction_count += len(re.findall(pattern, sentence))
    
    total_words = len(re.findall(r'\b\w+\b', text.lower()))
    
    if total_words == 0:
        return 1.0
    
    # Normalized contradiction score (inverted so higher = better)
    contradiction_density = (negation_count + contradiction_count) / total_words
    return max(0.0, 1.0 - contradiction_density * 10)  # Scale factor for reasonable range


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _count_syllables(word: str) -> int:
    """
    Count syllables in a word using a simple heuristic.
    """
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    # Handle silent e
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    # Every word has at least one syllable
    return max(1, syllable_count)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    sample_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter 
    of the alphabet at least once. However, it is not particularly complex in terms 
    of vocabulary or sentence structure. Therefore, we can use it as a baseline for 
    testing our readability metrics. Moreover, it demonstrates the importance of 
    having diverse sentence structures in text analysis.
    """
    
    print("=== READABILITY METRICS ===")
    print(f"Flesch Reading Ease: {flesch_reading_ease(sample_text):.2f}")
    print(f"Flesch-Kincaid Grade: {flesch_kincaid_grade_level(sample_text):.2f}")
    print(f"ARI: {automated_readability_index(sample_text):.2f}")
    print(f"Gunning Fog: {gunning_fog_index(sample_text):.2f}")
    print(f"SMOG: {smog_readability(sample_text):.2f}")
    
    print("\n=== VOCABULARY METRICS ===")
    print(f"Type-Token Ratio: {type_token_ratio(sample_text):.3f}")
    print(f"MATTR: {moving_average_ttr(sample_text):.3f}")
    print(f"Yule's K: {yule_k_characteristic(sample_text):.2f}")
    print(f"Avg Word Length: {average_word_length(sample_text):.2f}")
    print(f"Avg Syllables: {syllable_complexity(sample_text):.2f}")
    
    print("\n=== LOGICAL INTEGRITY METRICS ===")
    print(f"Logical Connector Density: {logical_connector_density(sample_text):.2f}")
    print(f"Argument Structure: {argument_structure_score(sample_text):.2f}")
    print(f"Coherence Score: {coherence_score(sample_text):.3f}")
    print(f"Contradiction Detection: {contradiction_detection_score(sample_text):.3f}")
