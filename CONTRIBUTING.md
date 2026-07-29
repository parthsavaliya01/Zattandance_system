# Contributing to Face Recognition Attendance System

First off, thank you for considering contributing to this project!

## How to Contribute

### 1. Reporting Bugs
- Search existing issues to see if it has already been reported.
- If not, open a new issue detailing:
  - Expected vs actual behavior.
  - Steps to reproduce.
  - System specs (CPU, GPU, RAM, OS, Python version, CUDA version).
  - Logs or error stack traces.

### 2. Suggesting Enhancements
- Open an issue explaining the proposed feature and why it would be beneficial.
- Describe how it should work or look.

### 3. Code Contributions (Pull Requests)
- Fork the repository.
- Create a new branch named after your feature/fix: `git checkout -b feature/your-feature-name`.
- Install development dependencies and linting/formatting tools.
- Write clear, commented code and ensure it complies with the codebase structure.
- Open a Pull Request pointing to the `main` branch.

## Code Style Guide

- Follow **PEP 8** standards for Python.
- Format your code with formatting tools like `black` or `yapf`.
- Keep functions modular and focused.
- Ensure sensitive parameters or hardcoded credentials are NEVER committed. Always externalize them to environment variables.
