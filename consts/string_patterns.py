# Language-specific comment patterns
COMMENT_PATTERNS = {
    'python': {
        'single': r'#.*$',
        'multi_start': r'"""',
        'multi_end': r'"""',
        'alt_multi_start': r"'''",
        'alt_multi_end': r"'''",
    },
    'javascript': {
        'single': r'//.*$',
        'multi_start': r'/\*',
        'multi_end': r'\*/',
    },
    'java': {
        'single': r'//.*$',
        'multi_start': r'/\*',
        'multi_end': r'\*/',
    },
    'c': {
        'single': r'//.*$',
        'multi_start': r'/\*',
        'multi_end': r'\*/',
    },
    'c++': {
        'single': r'//.*$',
        'multi_start': r'/\*',
        'multi_end': r'\*/',
    },
    'c#': {
        'single': r'//.*$',
        'multi_start': r'/\*',
        'multi_end': r'\*/',
    },
    'go': {
        'single': r'//.*$',
        'multi_start': r'/\*',
        'multi_end': r'\*/',
    },
}

# Boilerplate patterns by language
BOILERPLATE_PATTERNS = {
    'python': [
        r'^from\s+\S+\s+import\s+',  # from x import y
        r'^import\s+\S+',  # import x
        r'^if\s+__name__\s*==\s*["\']__main__["\']',  # if __name__ == "__main__"
        r'^\s*pass\s*$',  # pass statements
        r'^\s*def\s+__init__\s*\(\s*self\s*\)',  # empty __init__
        r'^\s*def\s+__str__\s*\(\s*self\s*\)',  # __str__
        r'^\s*def\s+__repr__\s*\(\s*self\s*\)',  # __repr__
        r'^\s*@property',  # property decorator
        r'^\s*@staticmethod',  # staticmethod
        r'^\s*@classmethod',  # classmethod
    ],
    'javascript': [
        r'^import\s+',  # import statements
        r'^export\s+(default\s+)?',  # export statements
        r'^const\s+\w+\s*=\s*require\(',  # require
        r'^module\.exports\s*=',  # module.exports
        r'^\s*constructor\s*\(',  # constructor
        r'^\s*get\s+\w+\s*\(',  # getters
        r'^\s*set\s+\w+\s*\(',  # setters
    ],
    'java': [
        r'^import\s+',  # import
        r'^package\s+',  # package
        r'^\s*@Override',  # @Override
        r'^\s*@Autowired',  # @Autowired
        r'^\s*public\s+\w+\s+get\w+\s*\(',  # getters
        r'^\s*public\s+void\s+set\w+\s*\(',  # setters
        r'^\s*private\s+\w+\s+\w+\s*;',  # private fields
    ],
    'c': [
        r'^#include\s+',  # includes
        r'^#define\s+',  # defines
        r'^#ifndef\s+',  # include guards
        r'^#ifdef\s+',  # ifdef
        r'^#endif',  # endif
        r'^#pragma\s+',  # pragmas
    ],
    'c++': [
        r'^#include\s+',  # includes
        r'^#define\s+',  # defines
        r'^#ifndef\s+',  # include guards
        r'^using\s+namespace\s+',  # using namespace
        r'^namespace\s+\w+\s*\{',  # namespace declaration
    ],
    'c#': [
        r'^using\s+',  # using
        r'^namespace\s+',  # namespace
        r'^\s*\[.*\]$',  # attributes
        r'^\s*public\s+\w+\s+\w+\s*\{\s*get;\s*set;\s*\}',  # auto properties
        r'^\s*get\s*\{',  # getter
        r'^\s*set\s*\{',  # setter
    ],
    'go': [
        r'^package\s+',  # package
        r'^import\s+',  # import
        r'^import\s*\(',  # import block
    ],
}

# Control flow keywords for complexity
CONTROL_FLOW_KEYWORDS = {
    'python': ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'and', 'or'],
    'javascript': ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch', 'finally', '&&', '||', '?'],
    'java': ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch', 'finally', '&&', '||', '?'],
    'c': ['if', 'else', 'for', 'while', 'do', 'switch', 'case', '&&', '||', '?'],
    'c++': ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'try', 'catch', '&&', '||', '?'],
    'c#': ['if', 'else', 'for', 'foreach', 'while', 'do', 'switch', 'case', 'try', 'catch', 'finally', '&&', '||', '?'],
    'go': ['if', 'else', 'for', 'switch', 'case', 'select', '&&', '||'],
}
