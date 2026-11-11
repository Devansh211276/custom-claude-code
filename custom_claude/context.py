"""Codebase indexer, .gitignore parser, and OPUS.md/CLAUDE.md project memory loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from custom_claude.tools import DEFAULT_IGNORES, is_ignored


class ProjectContext:
    """Manages repository context, directory trees, and project memory."""

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
        self.custom_ignores: Set[str] = self.load_gitignore()
        self.memory_file: Optional[Path] = self.find_memory_file()

    def load_gitignore(self) -> Set[str]:
        """Parse rules from .gitignore in project root."""
        ignores = set(DEFAULT_IGNORES)
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            ignores.add(line.rstrip("/"))
            except Exception:
                pass
        return ignores

    def find_memory_file(self) -> Optional[Path]:
        """Look for OPUS.md or CLAUDE.md in root."""
        for candidate in ["OPUS.md", "CLAUDE.md", "AGENTS.md"]:
            p = self.root_dir / candidate
            if p.exists() and p.is_file():
                return p
        return None

    def get_project_memory(self) -> str:
        """Load project memory contents or return default guidance."""
        if self.memory_file and self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8", errors="replace") as f:
                    return f"### Project Memory ({self.memory_file.name}):\n" + f.read()
            except Exception as exc:
                return f"[Warning: Could not read memory file {self.memory_file.name}: {exc}]"
        return "No OPUS.md or CLAUDE.md found. Run `/init` to generate one."

    def build_directory_tree(self, max_depth: int = 3, max_files: int = 80) -> str:
        """Generate a clean ASCII directory tree representation."""
        lines: List[str] = [f"{self.root_dir.name}/"]
        count = 0

        def _walk(directory: Path, prefix: str, depth: int) -> None:
            nonlocal count
            if depth > max_depth or count >= max_files:
                return

            try:
                entries = sorted(
                    list(directory.iterdir()),
                    key=lambda x: (not x.is_dir(), x.name.lower()),
                )
            except (PermissionError, FileNotFoundError):
                return

            # Filter ignored files
            valid_entries = [
                e for e in entries if not is_ignored(e, self.root_dir, self.custom_ignores)
            ]

            total = len(valid_entries)
            for i, entry in enumerate(valid_entries):
                if count >= max_files:
                    lines.append(f"{prefix}└── ... (tree truncated at {max_files} items)")
                    break

                is_last = (i == total - 1)
                connector = "└── " if is_last else "├── "
                sub_prefix = "    " if is_last else "│   "

                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    count += 1
                    _walk(entry, prefix + sub_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")
                    count += 1

        _walk(self.root_dir, "", 1)
        return "\n".join(lines)

    def scan_project_specs(self) -> Dict[str, str]:
        """Detect project technologies, package managers, and test runners."""
        specs: Dict[str, str] = {
            "type": "General Codebase",
            "test_runner": "pytest",
            "build_tool": "pip",
            "package_config": "",
        }

        # Check for Python
        if (self.root_dir / "pyproject.toml").exists():
            specs["type"] = "Python (pyproject.toml)"
            specs["test_runner"] = "pytest"
            specs["build_tool"] = "pip install -e ."
            specs["package_config"] = "pyproject.toml"
        elif (self.root_dir / "setup.py").exists():
            specs["type"] = "Python (setup.py)"
            specs["test_runner"] = "pytest"
            specs["build_tool"] = "pip install -e ."
            specs["package_config"] = "setup.py"
        elif (self.root_dir / "requirements.txt").exists():
            specs["type"] = "Python (requirements.txt)"
            specs["test_runner"] = "pytest"
            specs["build_tool"] = "pip install -r requirements.txt"
            specs["package_config"] = "requirements.txt"

        # Check for Node
        if (self.root_dir / "package.json").exists():
            specs["type"] = "Node.js / TypeScript"
            specs["test_runner"] = "npm test"
            specs["build_tool"] = "npm install"
            specs["package_config"] = "package.json"

        # Check for Rust
        if (self.root_dir / "Cargo.toml").exists():
            specs["type"] = "Rust"
            specs["test_runner"] = "cargo test"
            specs["build_tool"] = "cargo build"
            specs["package_config"] = "Cargo.toml"

        return specs

    def initialize_opus_md(self, force: bool = False) -> str:
        """Scan repository and generate OPUS.md context memory file."""
        target_path = self.root_dir / "OPUS.md"
        if target_path.exists() and not force:
            return f"OPUS.md already exists at {target_path}. Use force=True to overwrite."

        specs = self.scan_project_specs()
        tree_preview = self.build_directory_tree(max_depth=2, max_files=40)

        content = f"""# Project Memory & Architecture Guide (OPUS.md)

## 1. Project Overview
- **Project Root**: `{self.root_dir.name}`
- **Project Type**: {specs["type"]}
- **Primary Build / Install Command**: `{specs["build_tool"]}`
- **Test Command**: `{specs["test_runner"]}`

## 2. Directory Structure Preview
```text
{tree_preview}
```

## 3. Development Commands & Workflow
- **Run Tests**: `{specs["test_runner"]}`
- **Lint / Format**: Run formatting before submitting PRs
- **CLI Agent Usage**: Run `ccode` to launch Custom Claude terminal

## 4. Coding Standards & Guidelines
- Maintain modular architecture, explicit type hints, and full docstrings.
- Ensure all tests pass with zero errors before concluding tasks.
- Keep safety checks active when running shell commands or modifying files.
"""

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.memory_file = target_path
            return f"Successfully generated '{target_path.name}' with architecture & test specs."
        except Exception as exc:
            return f"Error creating OPUS.md: {exc}"

    def get_system_context(self) -> str:
        """Assemble full context injection string for the agent loop."""
        tree = self.build_directory_tree(max_depth=3, max_files=60)
        memory = self.get_project_memory()
        specs = self.scan_project_specs()

        context_str = f"""Current Working Directory: {self.root_dir}
Project Type: {specs['type']}
Test Command: {specs['test_runner']}

Repository File Tree:
{tree}

{memory}
"""
        return context_str
