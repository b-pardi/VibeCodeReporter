#! contains constants for programming files, used to filter git mining to exclude data files like images, serialized objects, text, json, csv, etx.

GENERAL_CODE_EXTENSIONS = {
    # Python
    ".py", ".pyi",

    # C / C++
    ".c", ".h", ".i",
    ".cpp", ".cc", ".cxx",
    ".hpp", ".hh", ".hxx",

    # Java / JVM
    ".java", ".kt", ".kts", ".scala",

    # Rust
    ".rs",

    # Go
    ".go",

    # C#
    ".cs",

    # Swift / Objective-C
    ".swift", ".m", ".mm",

    # Dart
    ".dart",

    # Julia
    ".jl",

    # Lua
    ".lua",

    # Zig
    ".zig",

    # Nim
    ".nim",

    # OCaml / ML
    ".ml", ".mli",

    # Haskell
    ".hs",

    # Fortran
    ".f", ".for", ".f90", ".f95",

    # R
    ".r",
}

WEB_CODE_EXTENSIONS = {
    # JavaScript / TypeScript
    ".js", ".mjs", ".cjs",
    ".ts", ".tsx",

    # JSX / TSX
    ".jsx",

    # HTML / templates
    ".html", ".htm",
    ".xhtml",

    # CSS / preprocessors
    ".css", ".scss", ".sass", ".less",

    # Web templates
    ".ejs", ".pug", ".jade",
    ".hbs", ".handlebars",
    ".jinja", ".j2",
}

ML_CODE_EXTENSIONS = {
    # Notebooks
    ".ipynb",

    # Model / config scripts
    ".cfg", ".ini",  # optional — include only if you want configs

    # MATLAB / Octave
    ".m", ".mat",

    # CUDA
    ".cu", ".cuh",

    # OpenCL
    ".cl",
}

SYSTEMS_CODE_EXTENSIONS = {
    # Assembly
    ".s", ".asm",

    # Linker / build
    ".ld",

    # Shell
    ".sh", ".bash", ".zsh",

    # Make / build systems
    ".make", ".mak",

    # Device trees
    ".dts", ".dtsi",
}

GAME_CODE_EXTENSIONS = {
    # Unity
    ".unity", ".prefab",

    # Unreal Engine
    ".uproject", ".uplugin",
    ".build.cs",

    # Godot
    ".gd",
}

BUILD_TOOLING_EXTENSIONS = {
    # CMake
    ".cmake",

    # Bazel
    ".bzl",

    # Gradle
    ".gradle",

    # Nix
    ".nix",

    # Docker
    ".dockerfile",
    "dockerfile",   # no extension

    # CI
    ".yml", ".yaml",
}

SCRIPTING_EXTENSIONS = {
    ".ps1",    # PowerShell
    ".bat",
    ".cmd",
    ".awk",
    ".sed",
}

CODE_EXTENSIONS = (
    GENERAL_CODE_EXTENSIONS
    | WEB_CODE_EXTENSIONS
    | ML_CODE_EXTENSIONS
    | SYSTEMS_CODE_EXTENSIONS
    | GAME_CODE_EXTENSIONS
    | BUILD_TOOLING_EXTENSIONS
    | SCRIPTING_EXTENSIONS
)

ALT_FNS = {
    "dockerfile",
    "makefile",
}