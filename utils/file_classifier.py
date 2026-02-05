"""
File type classification for extracted code.

Classifies files into categories: source, test, config, script, header, interface, etc.
"""

import re
from pathlib import Path


# File type definitions
FILE_TYPES = {
    'source': 'Main source code files',
    'test': 'Test files',
    'config': 'Configuration files',
    'script': 'Standalone scripts (entry points, CLI tools)',
    'header': 'Header/interface files (C/C++)',
    'interface': 'Interface/type definition files',
    'generated': 'Generated code files',
    'example': 'Example/sample code',
    'benchmark': 'Benchmark/performance test files',
    'migration': 'Database migration files',
    'unknown': 'Unable to classify',
}


# Path patterns for classification
PATH_PATTERNS = {
    'test': [
        r'tests?/',
        r'__tests__/',
        r'spec/',
        r'test_[^/]+$',
        r'[^/]+_test\.[^/]+$',
        r'[^/]+\.test\.[^/]+$',
        r'[^/]+\.spec\.[^/]+$',
        r'[^/]+Test\.(java|kt|scala)$',
        r'[^/]+Tests?\.(cs|fs)$',
    ],
    'config': [
        r'config/',
        r'configs?/',
        r'settings/',
        r'\.?[^/]*config[^/]*\.(py|js|ts|json|yaml|yml|toml)$',
        r'setup\.(py|cfg)$',
        r'pyproject\.toml$',
        r'package\.json$',
        r'tsconfig\.json$',
        r'webpack\.[^/]+\.js$',
        r'\.eslintrc',
        r'\.prettierrc',
        r'Makefile$',
        r'CMakeLists\.txt$',
        r'\.env(\.[^/]+)?$',
    ],
    'script': [
        r'scripts?/',
        r'bin/',
        r'tools?/',
        r'cli/',
        r'__main__\.py$',
        r'manage\.py$',
        r'setup\.py$',
        r'run[_-]?[^/]*\.(py|sh|js)$',
    ],
    'example': [
        r'examples?/',
        r'samples?/',
        r'demos?/',
        r'tutorials?/',
        r'quickstart/',
    ],
    'benchmark': [
        r'benchmarks?/',
        r'perf/',
        r'performance/',
        r'[^/]+_bench\.[^/]+$',
        r'[^/]+\.bench\.[^/]+$',
        r'Benchmark[^/]*\.(java|kt)$',
    ],
    'migration': [
        r'migrations?/',
        r'alembic/',
        r'db/migrate/',
        r'\d{4}[_-]?\d{2}[_-]?\d{2}.*\.(py|sql)$',  # Date-prefixed migrations
    ],
    'generated': [
        r'generated/',
        r'gen/',
        r'_gen\.[^/]+$',
        r'\.pb\.(go|py|java)$',  # Protobuf
        r'\.g\.dart$',  # Dart generated
    ],
    'header': [
        r'include/',
        r'headers?/',
        r'\.(h|hpp|hxx|hh)$',
    ],
    'interface': [
        r'interfaces?/',
        r'types?/',
        r'\.d\.ts$',  # TypeScript declarations
        r'I[A-Z][^/]+\.(cs|java)$',  # Interface naming convention
    ],
}

# Content patterns for additional classification
CONTENT_PATTERNS = {
    'test': [
        r'import\s+(unittest|pytest|nose)',
        r'from\s+(unittest|pytest|nose)',
        r'describe\s*\(',
        r'it\s*\(',
        r'test\s*\(',
        r'@Test\b',
        r'@pytest\.',
        r'assert\w*\s*\(',
        r'expect\s*\(',
        r'\.to(Equal|Be|Have)',
        r'class\s+\w*Test\w*',
        r'def\s+test_\w+',
        r'func\s+Test\w+',
    ],
    'script': [
        r'^#!.*\b(python|node|bash|sh|ruby|perl)\b',  # Shebang
        r'if\s+__name__\s*==\s*["\']__main__["\']',
        r'argparse\.ArgumentParser',
        r'click\.command',
        r'typer\.Typer',
        r'process\.argv',
        r'yargs\.',
        r'commander\.',
    ],
    'config': [
        r'^\s*[A-Z_]+\s*=',  # ALL_CAPS assignments (settings)
        r'settings\s*=\s*\{',
        r'config\s*=\s*\{',
        r'export\s+default\s*\{',  # Default export of config object
    ],
    'generated': [
        r'DO\s+NOT\s+(EDIT|MODIFY)',
        r'@generated',
        r'Generated\s+by',
        r'Auto-generated',
        r'THIS\s+FILE\s+IS\s+GENERATED',
    ],
}


def classify_file_type(file_path: str, content: str = "") -> str:
    """
    Classify a file into a type category based on path and content.

    Args:
        file_path: Path to the file (can be relative or absolute)
        content: Optional file content for content-based classification

    Returns:
        One of the FILE_TYPES keys
    """
    # Normalize path separators
    normalized_path = file_path.replace('\\', '/')

    # First check path patterns (more reliable)
    for file_type, patterns in PATH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized_path, re.IGNORECASE):
                return file_type

    # Then check content patterns if content is provided
    if content:
        # Check first 100 lines for efficiency
        header = '\n'.join(content.split('\n')[:100])

        for file_type, patterns in CONTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, header, re.MULTILINE | re.IGNORECASE):
                    return file_type

    # Default to source for code files
    ext = Path(file_path).suffix.lower()
    code_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.kt', '.scala',
        '.c', '.cpp', '.cc', '.cxx', '.cs', '.go', '.rs', '.rb', '.php',
        '.swift', '.m', '.mm', '.dart', '.lua', '.r', '.jl', '.hs',
    }

    if ext in code_extensions:
        return 'source'

    return 'unknown'


def get_file_category(file_type: str) -> str:
    """
    Get a broader category for a file type.

    Categories:
    - 'code': source, header, interface
    - 'test': test, benchmark
    - 'config': config, migration
    - 'other': script, example, generated, unknown
    """
    categories = {
        'source': 'code',
        'header': 'code',
        'interface': 'code',
        'test': 'test',
        'benchmark': 'test',
        'config': 'config',
        'migration': 'config',
        'script': 'other',
        'example': 'other',
        'generated': 'other',
        'unknown': 'other',
    }
    return categories.get(file_type, 'other')


def is_production_code(file_type: str) -> bool:
    """Check if file type represents production code (not test/example/config)."""
    return file_type in {'source', 'header', 'interface'}


def classify_by_extension(file_path: str) -> dict:
    """
    Get metadata about a file based on its extension.

    Returns dict with:
    - language: Programming language
    - is_code: Whether it's a code file
    - category: General category
    """
    ext = Path(file_path).suffix.lower()

    extension_info = {
        # Python
        '.py': {'language': 'python', 'is_code': True, 'category': 'source'},
        '.pyi': {'language': 'python', 'is_code': True, 'category': 'interface'},
        '.pyx': {'language': 'python', 'is_code': True, 'category': 'source'},

        # JavaScript/TypeScript
        '.js': {'language': 'javascript', 'is_code': True, 'category': 'source'},
        '.jsx': {'language': 'javascript', 'is_code': True, 'category': 'source'},
        '.mjs': {'language': 'javascript', 'is_code': True, 'category': 'source'},
        '.ts': {'language': 'typescript', 'is_code': True, 'category': 'source'},
        '.tsx': {'language': 'typescript', 'is_code': True, 'category': 'source'},
        '.d.ts': {'language': 'typescript', 'is_code': True, 'category': 'interface'},

        # Java/JVM
        '.java': {'language': 'java', 'is_code': True, 'category': 'source'},
        '.kt': {'language': 'kotlin', 'is_code': True, 'category': 'source'},
        '.scala': {'language': 'scala', 'is_code': True, 'category': 'source'},
        '.groovy': {'language': 'groovy', 'is_code': True, 'category': 'source'},

        # C/C++
        '.c': {'language': 'c', 'is_code': True, 'category': 'source'},
        '.h': {'language': 'c', 'is_code': True, 'category': 'header'},
        '.cpp': {'language': 'c++', 'is_code': True, 'category': 'source'},
        '.cc': {'language': 'c++', 'is_code': True, 'category': 'source'},
        '.cxx': {'language': 'c++', 'is_code': True, 'category': 'source'},
        '.hpp': {'language': 'c++', 'is_code': True, 'category': 'header'},
        '.hh': {'language': 'c++', 'is_code': True, 'category': 'header'},
        '.hxx': {'language': 'c++', 'is_code': True, 'category': 'header'},

        # C#
        '.cs': {'language': 'c#', 'is_code': True, 'category': 'source'},

        # Go
        '.go': {'language': 'go', 'is_code': True, 'category': 'source'},

        # Rust
        '.rs': {'language': 'rust', 'is_code': True, 'category': 'source'},

        # Ruby
        '.rb': {'language': 'ruby', 'is_code': True, 'category': 'source'},

        # PHP
        '.php': {'language': 'php', 'is_code': True, 'category': 'source'},

        # Swift
        '.swift': {'language': 'swift', 'is_code': True, 'category': 'source'},

        # Config/Data
        '.json': {'language': 'json', 'is_code': False, 'category': 'config'},
        '.yaml': {'language': 'yaml', 'is_code': False, 'category': 'config'},
        '.yml': {'language': 'yaml', 'is_code': False, 'category': 'config'},
        '.toml': {'language': 'toml', 'is_code': False, 'category': 'config'},
        '.xml': {'language': 'xml', 'is_code': False, 'category': 'config'},
        '.ini': {'language': 'ini', 'is_code': False, 'category': 'config'},
    }

    return extension_info.get(ext, {
        'language': 'unknown',
        'is_code': False,
        'category': 'unknown'
    })
