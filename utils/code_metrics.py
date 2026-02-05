"""
Code metrics computation for extracted files.

Includes LOC, comment density, boilerplate detection, and complexity metrics.
"""

import re
from typing import Optional

from consts.string_patterns import *


def compute_loc(content: Optional[str]) -> int:
    """Compute lines of code (non-empty, non-whitespace lines)."""
    if not content:
        return 0
    lines = content.split('\n')
    return sum(1 for line in lines if line.strip())


def compute_comment_density(content: str, language: str) -> float:
    """
    Compute the ratio of comment lines to total non-empty lines.
    Returns a value between 0 and 1.
    """
    if not content:
        return 0.0

    language = language.lower()
    patterns = COMMENT_PATTERNS.get(language, COMMENT_PATTERNS['python'])

    lines = content.split('\n')
    total_lines = 0
    comment_lines = 0
    in_multiline = False
    multi_end = patterns.get('multi_end', '')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        total_lines += 1

        # Check if we're in a multiline comment
        if in_multiline:
            comment_lines += 1
            if multi_end and multi_end in stripped:
                in_multiline = False
            continue

        # Check for single-line comment
        single_pattern = patterns.get('single', '')
        if single_pattern and re.match(single_pattern, stripped):
            comment_lines += 1
            continue

        # Check for multiline comment start
        multi_start = patterns.get('multi_start', '')
        if multi_start and multi_start in stripped:
            comment_lines += 1
            # Check if it also ends on this line
            if not (multi_end and multi_end in stripped.split(multi_start, 1)[-1]):
                in_multiline = True
            continue

        # Check alternate multiline (Python docstrings)
        alt_start = patterns.get('alt_multi_start', '')
        alt_end = patterns.get('alt_multi_end', '')
        if alt_start and alt_start in stripped:
            comment_lines += 1
            count = stripped.count(alt_start)
            if count == 1:
                in_multiline = True
                multi_end = alt_end

    return comment_lines / total_lines if total_lines > 0 else 0.0


def compute_boilerplate_score(content: str, language: str) -> float:
    """
    Compute a boilerplate score (0-1) indicating how much of the code is boilerplate.
    Higher values indicate more boilerplate.
    """
    if not content:
        return 0.0

    language = language.lower()
    patterns = BOILERPLATE_PATTERNS.get(language, [])

    if not patterns:
        return 0.0

    lines = content.split('\n')
    total_lines = 0
    boilerplate_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        total_lines += 1

        for pattern in patterns:
            if re.match(pattern, stripped, re.MULTILINE):
                boilerplate_lines += 1
                break

    return boilerplate_lines / total_lines if total_lines > 0 else 0.0


def compute_complexity_metrics(content: str, language: str) -> dict:
    """
    Compute complexity metrics for code.

    Returns:
        dict with keys:
        - cyclomatic: Cyclomatic complexity estimate
        - cognitive: Cognitive complexity estimate
        - max_nesting: Maximum nesting depth
    """
    if not content:
        return {'cyclomatic': None, 'cognitive': None, 'max_nesting': None}

    language = language.lower()

    # For Python, try to use radon if available
    if language == 'python':
        try:
            from radon.complexity import cc_visit
            from radon.metrics import mi_visit

            blocks = cc_visit(content)
            if blocks:
                cyclomatic = sum(block.complexity for block in blocks)
            else:
                cyclomatic = 1
        except Exception:
            cyclomatic = _estimate_cyclomatic(content, language)
    else:
        cyclomatic = _estimate_cyclomatic(content, language)

    cognitive = _estimate_cognitive_complexity(content, language)
    max_nesting = _compute_max_nesting(content, language)

    return {
        'cyclomatic': cyclomatic,
        'cognitive': cognitive,
        'max_nesting': max_nesting,
    }


def _estimate_cyclomatic(content: str, language: str) -> int:
    """
    Estimate cyclomatic complexity by counting decision points.
    CC = E - N + 2P, but simplified as 1 + decision_points for single function.
    """
    keywords = CONTROL_FLOW_KEYWORDS.get(language, CONTROL_FLOW_KEYWORDS['python'])

    complexity = 1  # Base complexity

    for keyword in keywords:
        if keyword in ['&&', '||', '?']:
            # Count operators
            complexity += content.count(keyword)
        else:
            # Count keyword occurrences (word boundaries)
            pattern = r'\b' + keyword + r'\b'
            complexity += len(re.findall(pattern, content))

    return complexity


def _estimate_cognitive_complexity(content: str, language: str) -> int:
    """
    Estimate cognitive complexity.
    Cognitive complexity adds penalties for nesting and certain constructs.
    """
    keywords = CONTROL_FLOW_KEYWORDS.get(language, CONTROL_FLOW_KEYWORDS['python'])

    complexity = 0
    lines = content.split('\n')
    nesting_level = 0

    # Simplified nesting detection based on indentation
    prev_indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Calculate indentation
        indent = len(line) - len(line.lstrip())

        # Update nesting level based on indentation changes
        if indent > prev_indent:
            nesting_level += 1
        elif indent < prev_indent:
            nesting_level = max(0, nesting_level - 1)

        prev_indent = indent

        # Add complexity for control flow with nesting penalty
        for keyword in keywords:
            if keyword in ['&&', '||']:
                if keyword in stripped:
                    complexity += 1
            elif keyword == '?':
                complexity += stripped.count('?')
            else:
                pattern = r'\b' + keyword + r'\b'
                matches = len(re.findall(pattern, stripped))
                if matches > 0:
                    # Add 1 for the construct + nesting level as penalty
                    complexity += matches * (1 + nesting_level)

    return complexity


def _compute_max_nesting(content: str, language: str) -> int:
    """
    Compute maximum nesting depth.
    Uses brace counting for C-like languages, indentation for Python.
    """
    if language.lower() == 'python':
        return _compute_max_nesting_indent(content)
    else:
        return _compute_max_nesting_braces(content)


def _compute_max_nesting_indent(content: str) -> int:
    """Compute max nesting by indentation (for Python)."""
    lines = content.split('\n')
    max_indent = 0

    for line in lines:
        if not line.strip():
            continue

        # Count leading spaces/tabs
        indent = 0
        for char in line:
            if char == ' ':
                indent += 1
            elif char == '\t':
                indent += 4  # Treat tab as 4 spaces
            else:
                break

        # Convert to nesting level (assume 4-space indent)
        nesting = indent // 4
        max_indent = max(max_indent, nesting)

    return max_indent


def _compute_max_nesting_braces(content: str) -> int:
    """Compute max nesting by brace counting (for C-like languages)."""
    max_depth = 0
    current_depth = 0

    # Simple state machine - not perfect but good enough
    in_string = False
    in_char = False
    in_comment = False
    prev_char = ''

    for char in content:
        # Handle comments
        if not in_string and not in_char:
            if prev_char == '/' and char == '*':
                in_comment = True
            elif prev_char == '*' and char == '/' and in_comment:
                in_comment = False
            elif prev_char == '/' and char == '/':
                # Skip to end of line (simplified)
                pass

        if in_comment:
            prev_char = char
            continue

        # Handle strings
        if char == '"' and prev_char != '\\' and not in_char:
            in_string = not in_string
        elif char == "'" and prev_char != '\\' and not in_string:
            in_char = not in_char

        if in_string or in_char:
            prev_char = char
            continue

        # Count braces
        if char == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth = max(0, current_depth - 1)

        prev_char = char

    return max_depth
