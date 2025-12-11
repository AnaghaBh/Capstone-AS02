#!/usr/bin/env python3
"""
Dataset Preprocessing Script for Misinformation Classifier

This script performs comprehensive text cleaning and preprocessing:
1. Remove duplicate texts
2. Normalize whitespace
3. Handle encoding issues and weird characters
4. Normalize quotes and apostrophes
5. Clean special characters
6. Validate data integrity

Usage:
    python scripts/preprocess_dataset.py --input ../Full_Final/Final.json --output ../Full_Final/Final_cleaned.json
"""

import json
import re
import argparse
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their canonical form.
    Converts characters like 'é' to 'e', handles various Unicode issues.
    """
    # Normalize to NFKC form (compatibility decomposition, then canonical composition)
    text = unicodedata.normalize('NFKC', text)
    return text


def normalize_quotes(text: str) -> str:
    """
    Normalize various quote styles to standard ASCII quotes.
    """
    # Single quotes and apostrophes
    single_quotes = [''', ''', '‛', '‚', '`', '´', 'ʼ', 'ʻ', 'ˈ', 'ˊ', 'ˋ']
    for q in single_quotes:
        text = text.replace(q, "'")
    
    # Double quotes
    double_quotes = ['"', '"', '„', '‟', '«', '»', '〝', '〞', '❝', '❞']
    for q in double_quotes:
        text = text.replace(q, '"')
    
    return text


def normalize_dashes(text: str) -> str:
    """
    Normalize various dash types to standard hyphen or em-dash.
    """
    # En-dash, em-dash, and other dashes to standard hyphen for compound words
    dashes = ['–', '—', '―', '‐', '‑', '‒', '⁃', '−']
    for d in dashes:
        text = text.replace(d, '-')
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace characters and remove extra spaces.
    """
    # Replace various whitespace characters with regular space
    whitespace_chars = [
        '\u00A0',  # Non-breaking space
        '\u2002',  # En space
        '\u2003',  # Em space
        '\u2004',  # Three-per-em space
        '\u2005',  # Four-per-em space
        '\u2006',  # Six-per-em space
        '\u2007',  # Figure space
        '\u2008',  # Punctuation space
        '\u2009',  # Thin space
        '\u200A',  # Hair space
        '\u200B',  # Zero-width space
        '\u202F',  # Narrow no-break space
        '\u205F',  # Medium mathematical space
        '\u3000',  # Ideographic space
        '\t',      # Tab
        '\r',      # Carriage return
        '\n',      # Newline
    ]
    
    for ws in whitespace_chars:
        text = text.replace(ws, ' ')
    
    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_special_characters(text: str) -> str:
    """
    Remove or replace problematic special characters while preserving meaning.
    """
    # Remove zero-width characters
    zero_width = ['\u200B', '\u200C', '\u200D', '\uFEFF', '\u00AD']
    for zw in zero_width:
        text = text.replace(zw, '')
    
    # Replace ellipsis character with three dots
    text = text.replace('…', '...')
    
    # Replace bullet points with hyphen
    bullets = ['•', '●', '○', '◦', '▪', '▫', '‣']
    for b in bullets:
        text = text.replace(b, '-')
    
    # Replace common symbols
    text = text.replace('™', '')
    text = text.replace('®', '')
    text = text.replace('©', '')
    text = text.replace('°', ' degrees ')
    text = text.replace('×', 'x')
    text = text.replace('÷', '/')
    
    return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters that might cause issues.
    """
    # Remove control characters except common whitespace
    cleaned = ''.join(
        char for char in text 
        if unicodedata.category(char) != 'Cc' or char in [' ', '\n', '\t']
    )
    return cleaned


def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Main text cleaning function that applies all normalizations.
    
    Args:
        text: Input text to clean
        aggressive: If True, converts to ASCII only (removes accented chars)
    
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Remove control characters
    text = remove_control_characters(text)
    
    # Step 2: Normalize Unicode
    text = normalize_unicode(text)
    
    # Step 3: Normalize quotes
    text = normalize_quotes(text)
    
    # Step 4: Normalize dashes
    text = normalize_dashes(text)
    
    # Step 5: Clean special characters
    text = clean_special_characters(text)
    
    # Step 6: Normalize whitespace (do this last)
    text = normalize_whitespace(text)
    
    # Optional: Convert to ASCII only (aggressive mode)
    if aggressive:
        # This removes accented characters by converting to closest ASCII
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = normalize_whitespace(text)  # Clean up again after ASCII conversion
    
    return text


def remove_duplicates(data: list, key: str = 'text') -> tuple:
    """
    Remove duplicate entries based on a key field.
    
    Returns:
        (unique_data, duplicate_count, duplicate_examples)
    """
    seen = set()
    unique_data = []
    duplicates = []
    
    for item in data:
        text = item.get(key, '')
        # Normalize for comparison (lowercase, stripped)
        normalized = text.lower().strip()
        
        if normalized not in seen:
            seen.add(normalized)
            unique_data.append(item)
        else:
            duplicates.append(text[:100])  # Keep first 100 chars as example
    
    return unique_data, len(duplicates), duplicates[:10]  # Return first 10 examples


def validate_labels(data: list) -> tuple:
    """
    Validate that all labels are 0 or 1.
    
    Returns:
        (valid_data, invalid_count, issues)
    """
    label_fields = [
        'framework1_feature1', 
        'framework1_feature2',
        'framework2_feature1', 
        'framework2_feature2', 
        'framework2_feature3'
    ]
    
    valid_data = []
    issues = []
    
    for item in data:
        is_valid = True
        for field in label_fields:
            value = item.get(field)
            
            # Convert to int if needed
            if isinstance(value, float):
                item[field] = int(value)
                value = item[field]
            elif isinstance(value, str):
                try:
                    item[field] = int(value)
                    value = item[field]
                except ValueError:
                    is_valid = False
                    issues.append(f"ID {item.get('id')}: {field} = '{value}' (not a number)")
                    continue
            
            if value not in [0, 1]:
                is_valid = False
                issues.append(f"ID {item.get('id')}: {field} = {value} (not 0 or 1)")
        
        if is_valid:
            valid_data.append(item)
    
    return valid_data, len(data) - len(valid_data), issues[:20]


def reassign_ids(data: list) -> list:
    """
    Reassign sequential IDs starting from 1.
    """
    for i, item in enumerate(data, start=1):
        item['id'] = i
    return data


def analyze_dataset(data: list) -> dict:
    """
    Generate statistics about the dataset.
    """
    texts = [item.get('text', '') for item in data]
    
    stats = {
        'total_samples': len(data),
        'text_stats': {
            'min_words': min(len(t.split()) for t in texts) if texts else 0,
            'max_words': max(len(t.split()) for t in texts) if texts else 0,
            'avg_words': sum(len(t.split()) for t in texts) / len(texts) if texts else 0,
            'min_chars': min(len(t) for t in texts) if texts else 0,
            'max_chars': max(len(t) for t in texts) if texts else 0,
            'avg_chars': sum(len(t) for t in texts) / len(texts) if texts else 0,
        },
        'label_distribution': {},
        'topics': dict(Counter(item.get('topic', 'unknown') for item in data)),
        'non_ascii_count': sum(1 for t in texts if any(ord(c) > 127 for c in t)),
    }
    
    # Label distribution
    label_names = {
        'framework1_feature1': 'central_route',
        'framework1_feature2': 'peripheral_route',
        'framework2_feature1': 'naturalness_bias',
        'framework2_feature2': 'availability_bias',
        'framework2_feature3': 'illusory_correlation',
    }
    
    for field, name in label_names.items():
        positive = sum(1 for item in data if item.get(field) == 1)
        stats['label_distribution'][name] = {
            'positive': positive,
            'negative': len(data) - positive,
            'positive_ratio': round(positive / len(data) * 100, 1) if data else 0
        }
    
    return stats


def preprocess_dataset(
    input_path: str,
    output_path: str = None,
    aggressive_cleaning: bool = False,
    remove_dups: bool = True,
    reassign_sequential_ids: bool = True
) -> dict:
    """
    Main preprocessing function.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to save cleaned JSON (if None, doesn't save)
        aggressive_cleaning: If True, converts to ASCII only
        remove_dups: If True, removes duplicate texts
        reassign_sequential_ids: If True, reassigns IDs from 1 to N
    
    Returns:
        Dictionary with preprocessing results and statistics
    """
    results = {
        'input_file': input_path,
        'output_file': output_path,
        'operations': [],
        'before_stats': {},
        'after_stats': {},
    }
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading data from: {input_path}")
    print('='*60)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    print(f"Loaded {original_count} samples")
    
    results['before_stats'] = analyze_dataset(data)
    
    # Step 1: Clean text
    print(f"\n[1/4] Cleaning text...")
    texts_with_issues = 0
    for item in data:
        original = item.get('text', '')
        cleaned = clean_text(original, aggressive=aggressive_cleaning)
        if original != cleaned:
            texts_with_issues += 1
        item['text'] = cleaned
    
    print(f"      Cleaned {texts_with_issues} texts with issues")
    results['operations'].append(f"Text cleaning: {texts_with_issues} texts modified")
    
    # Step 2: Remove duplicates
    if remove_dups:
        print(f"\n[2/4] Removing duplicates...")
        data, dup_count, dup_examples = remove_duplicates(data)
        print(f"      Removed {dup_count} duplicates")
        if dup_examples:
            print(f"      Examples of removed duplicates:")
            for ex in dup_examples[:3]:
                print(f"        - \"{ex[:80]}...\"")
        results['operations'].append(f"Duplicate removal: {dup_count} removed")
    else:
        print(f"\n[2/4] Skipping duplicate removal")
    
    # Step 3: Validate labels
    print(f"\n[3/4] Validating labels...")
    data, invalid_count, issues = validate_labels(data)
    print(f"      Found {invalid_count} samples with invalid labels")
    if issues:
        print(f"      Issues:")
        for issue in issues[:5]:
            print(f"        - {issue}")
    results['operations'].append(f"Label validation: {invalid_count} invalid samples removed")
    
    # Step 4: Reassign IDs
    if reassign_sequential_ids:
        print(f"\n[4/4] Reassigning sequential IDs...")
        data = reassign_ids(data)
        print(f"      IDs reassigned from 1 to {len(data)}")
        results['operations'].append(f"ID reassignment: 1 to {len(data)}")
    else:
        print(f"\n[4/4] Skipping ID reassignment")
    
    # Final statistics
    results['after_stats'] = analyze_dataset(data)
    
    # Summary
    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print('='*60)
    print(f"Original samples:  {original_count}")
    print(f"Final samples:     {len(data)}")
    print(f"Samples removed:   {original_count - len(data)}")
    print(f"\nLabel Distribution (after cleaning):")
    for label, stats in results['after_stats']['label_distribution'].items():
        print(f"  {label}: {stats['positive']} positive ({stats['positive_ratio']}%)")
    print(f"\nNon-ASCII texts remaining: {results['after_stats']['non_ascii_count']}")
    
    # Save if output path provided
    if output_path:
        print(f"\n{'='*60}")
        print(f"Saving cleaned data to: {output_path}")
        print('='*60)
        
        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} samples")
        results['saved'] = True
    else:
        results['saved'] = False
    
    results['final_data'] = data
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess misinformation dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing
  python preprocess_dataset.py --input ../Full_Final/Final.json --output ../Full_Final/Final_cleaned.json
  
  # Aggressive cleaning (ASCII only)
  python preprocess_dataset.py --input ../Full_Final/Final.json --output ../Full_Final/Final_cleaned.json --aggressive
  
  # Keep duplicates (not recommended)
  python preprocess_dataset.py --input ../Full_Final/Final.json --output ../Full_Final/Final_cleaned.json --keep-duplicates
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output JSON file (if not specified, will add _cleaned suffix)'
    )
    
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Use aggressive cleaning (convert to ASCII only)'
    )
    
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Keep duplicate texts (not recommended)'
    )
    
    parser.add_argument(
        '--keep-ids',
        action='store_true',
        help='Keep original IDs (do not reassign sequentially)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving output file'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.dry_run:
        output_path = None
    elif args.output:
        output_path = args.output
    else:
        # Add _cleaned suffix
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    
    # Run preprocessing
    results = preprocess_dataset(
        input_path=args.input,
        output_path=output_path,
        aggressive_cleaning=args.aggressive,
        remove_dups=not args.keep_duplicates,
        reassign_sequential_ids=not args.keep_ids
    )
    
    print(f"\n✅ Preprocessing complete!")
    
    if results['saved']:
        print(f"   Output saved to: {results['output_file']}")
    else:
        print(f"   Dry run - no file saved")


if __name__ == '__main__':
    main()
