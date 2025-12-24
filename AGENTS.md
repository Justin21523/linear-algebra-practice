# Repository Guidelines

## Project Structure & Module Organization

- Chapters live at the repo root (e.g., `01-vectors-and-matrices/` … `06-eigenvalues-and-eigenvectors/`).
- Each chapter contains unit folders like `NN-topic-name/` with a `README.md` plus language subfolders (commonly `python/`, `c/`, `cpp/`, `java/`, `javascript/`, `csharp/`).
- Python units often have paired implementations: `*_manual.py` (from-scratch) and `*_numpy.py` (uses NumPy).

## Build, Test, and Development Commands

This repo is a collection of standalone examples (no single global build). Run code from the unit folder you’re working in:

- Python:
  ```bash
  cd 04-orthogonality-and-least-squares/02-projections/python
  python projection_manual.py
  ```
  NumPy variants require `numpy` installed.
- C / C++ (see each file header for exact flags):
  ```bash
  gcc -std=c99 -O2 inner_product.c -o inner_product -lm && ./inner_product
  g++ -std=c++17 -O2 inner_product.cpp -o inner_product && ./inner_product
  ```
- Java / JavaScript:
  ```bash
  javac InnerProduct.java && java InnerProduct
  node inner_product.js
  ```
- C#: compile standalone files with `csc`, or create a small `dotnet new console` project if you prefer `dotnet run`.

## Coding Style & Naming Conventions

- Indentation: 4 spaces (match existing files). Keep bilingual explanations (中文 + English terms) and the “what this program demonstrates” header style.
- Comments: for new/updated source files, keep detailed English comments on most lines (aim for >90% line coverage).
- Naming: preserve `NN-topic-name/` directories and existing file patterns (e.g., `*_manual.*`, `*_numpy.py`, `snake_case` in Python, `CamelCase` in Java/C#).
- Docs: for every new/changed unit, add a matching Traditional-Chinese explanation doc under `docs/implementations/<chapter>/<unit>/README.md` (see `docs/README.md`).
- Keep examples self-contained and avoid adding heavy dependencies; if you add one, document it in the closest `README.md`.

## Testing Guidelines

- There is no centralized test runner today. Validate changes by running the affected script(s) and checking output against the unit `README.md`.
- When a manual and library-backed version both exist, they should produce equivalent results (allowing for small floating-point tolerances).

## Commit & Pull Request Guidelines

- Git history is minimal (e.g., `first commit`), so there’s no established commit convention yet. Prefer short, imperative subjects; optionally prefix with the chapter (e.g., `04: fix projection edge case`).
- PRs should explain the concept/topic, list affected paths, and include run instructions plus sample output (paste or screenshot) for at least one language implementation.
