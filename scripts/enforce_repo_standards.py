#!/usr/bin/env python3  # EN: Use the system Python interpreter to run this maintenance script.

"""  # EN: Start of the module docstring explaining what this script does.
Repository maintenance helpers: line comments + unit docs.  # EN: Summarize the two main responsibilities of this script.

This script is intentionally dependency-free.  # EN: Avoid external packages so contributors can run it easily.
"""  # EN: End of the module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations for older Python versions.

import argparse  # EN: Parse command-line flags for this script.
import re  # EN: Provide regular expressions for lightweight parsing.
from dataclasses import dataclass  # EN: Use dataclasses for small structured records.
from pathlib import Path  # EN: Use pathlib for safe, readable filesystem paths.
from typing import Iterable  # EN: Type helper for iterable return types.


SOURCE_EXTENSIONS = {".py", ".c", ".cpp", ".java", ".js", ".cs"}  # EN: File extensions treated as source code in this repo.
SKIP_DIR_NAMES = {".git", "docs"}  # EN: Directories to skip during source scanning to avoid rewriting generated docs.
GENERATED_MARKER = "EN:"  # EN: Marker used to detect previously generated inline English comments.


@dataclass(frozen=True)  # EN: Declare an immutable record for per-language comment syntax.
class CommentStyle:  # EN: Hold language-specific comment tokens and behaviors.
    token: str  # EN: The inline comment token (e.g., "#" or "//").
    marker: str  # EN: The unique marker prefix used in generated comments.


PY_STYLE = CommentStyle(token="#", marker="# EN:")  # EN: Python uses hash comments.
CLIKE_STYLE = CommentStyle(token="//", marker="// EN:")  # EN: C-like languages use double-slash comments.


def main() -> int:  # EN: Provide a CLI entrypoint returning a process exit code.
    parser = argparse.ArgumentParser(  # EN: Create a CLI parser for this script.
        description="Enforce repo conventions (inline EN comments + unit docs).",  # EN: Explain the script purpose.
    )  # EN: Finish constructing the ArgumentParser.
    parser.add_argument(  # EN: Add a flag to enable writing changes to disk.
        "--write",  # EN: Name of the write flag.
        action="store_true",  # EN: Treat presence of the flag as True.
        help="Write changes to files (default: dry-run).",  # EN: Explain default behavior and what the flag does.
    )  # EN: Finish defining the --write argument.
    parser.add_argument(  # EN: Add a flag controlling doc generation behavior.
        "--docs",  # EN: Name of the docs flag.
        action="store_true",  # EN: Treat presence of the flag as True.
        help="Generate docs/implementations/<chapter>/<unit>/README.md files.",  # EN: Explain which docs are generated.
    )  # EN: Finish defining the --docs argument.
    parser.add_argument(  # EN: Add a flag controlling source annotation behavior.
        "--annotate",  # EN: Name of the annotate flag.
        action="store_true",  # EN: Treat presence of the flag as True.
        help="Annotate source files with inline English comments.",  # EN: Explain which files will be annotated.
    )  # EN: Finish defining the --annotate argument.
    args = parser.parse_args()  # EN: Parse CLI arguments from sys.argv.

    repo_root = Path(__file__).resolve().parents[1]  # EN: Derive the repository root from the scripts/ folder.

    annotate_enabled = args.annotate or (not args.docs)  # EN: Default to annotation unless docs-only is requested.
    docs_enabled = args.docs  # EN: Store whether unit doc generation is enabled.

    changed_files: list[Path] = []  # EN: Track which files were modified to report summary information.

    if annotate_enabled:  # EN: Only annotate source files when enabled.
        for file_path in iter_source_files(repo_root):  # EN: Walk every source file under the repo root.
            updated_text = annotate_text(file_path, file_path.read_text(encoding="utf-8"))  # EN: Produce annotated text.
            if updated_text is None:  # EN: Skip files that do not need updates.
                continue  # EN: Continue scanning the next file.
            if args.write:  # EN: Only write when --write is provided.
                file_path.write_text(updated_text, encoding="utf-8")  # EN: Persist the updated source file.
            changed_files.append(file_path)  # EN: Record this file as changed (or would-change in dry-run).

    if docs_enabled:  # EN: Only generate unit docs when enabled.
        for doc_path in generate_unit_docs(repo_root=repo_root, write=args.write):  # EN: Create/refresh unit docs and yield paths.
            changed_files.append(doc_path)  # EN: Record docs that were written (or would be written in dry-run).

    if changed_files:  # EN: If any changes were made (or would be made), print a short summary.
        mode = "WRITE" if args.write else "DRY-RUN"  # EN: Decide the reporting mode string.
        print(f"[{mode}] updated {len(changed_files)} files")  # EN: Print the count of touched files.
    else:  # EN: If nothing changed, tell the user explicitly.
        print("No changes needed.")  # EN: Print a friendly no-op message.

    return 0  # EN: Return success.


def iter_source_files(repo_root: Path) -> Iterable[Path]:  # EN: Yield all source files we want to annotate.
    for path in repo_root.rglob("*"):  # EN: Recursively scan all files and directories under the repo root.
        if any(part in SKIP_DIR_NAMES for part in path.parts):  # EN: Skip special directories (e.g., docs/, .git/).
            continue  # EN: Ignore this path and continue scanning.
        if path.is_dir():  # EN: Only yield actual files.
            continue  # EN: Skip directories.
        if path.suffix not in SOURCE_EXTENSIONS:  # EN: Only include recognized source code extensions.
            continue  # EN: Skip non-source files.
        if path.name == "enforce_repo_standards.py":  # EN: Avoid rewriting this script during the same run.
            continue  # EN: Skip self to keep the script stable.
        yield path  # EN: Yield the file path for processing.


def comment_style_for_path(path: Path) -> CommentStyle:  # EN: Choose the comment style based on file extension.
    if path.suffix == ".py":  # EN: Python uses hash comments.
        return PY_STYLE  # EN: Return Python comment style.
    return CLIKE_STYLE  # EN: Default to C-like style for the remaining languages.


def annotate_text(path: Path, text: str) -> str | None:  # EN: Annotate a file’s content; return None if unchanged.
    style = comment_style_for_path(path)  # EN: Pick the correct inline comment syntax.
    lines = text.splitlines(keepends=True)  # EN: Preserve existing newline characters while iterating.

    in_triple = False  # EN: Track whether we are inside a Python triple-quoted string.
    triple_delim: str | None = None  # EN: Track which triple-quote delimiter opened the current string.

    changed = False  # EN: Track whether we modified any line.
    out_lines: list[str] = []  # EN: Collect rewritten lines for output.

    for raw_line in lines:  # EN: Process each original line in order.
        line_no_nl = raw_line[:-1] if raw_line.endswith("\n") else raw_line  # EN: Remove newline for easier editing.
        newline = "\n" if raw_line.endswith("\n") else ""  # EN: Preserve whether this line originally ended with newline.

        stripped = line_no_nl.strip()  # EN: Compute a whitespace-trimmed version for classification.
        if not stripped:  # EN: Leave empty/whitespace-only lines unchanged.
            out_lines.append(raw_line)  # EN: Preserve blank lines exactly.
            continue  # EN: Move to the next line.

        if style is PY_STYLE:  # EN: Python requires special handling to avoid touching docstrings.
            in_triple, triple_delim = update_triple_quote_state(  # EN: Update whether we are currently inside triple quotes.
                stripped=stripped,  # EN: Pass the trimmed line for simple detection.
                in_triple=in_triple,  # EN: Provide the current state.
                triple_delim=triple_delim,  # EN: Provide the current delimiter state.
            )  # EN: Finish updating the triple-quote state.
            if in_triple:  # EN: Avoid annotating lines inside triple-quoted strings.
                out_lines.append(raw_line)  # EN: Preserve docstring/multiline string lines unchanged.
                continue  # EN: Move to the next line.

        if is_comment_only_line(stripped=stripped, style=style):  # EN: Skip pure comment lines.
            out_lines.append(raw_line)  # EN: Preserve existing comment lines without rewriting.
            continue  # EN: Move to the next line.

        if stripped.endswith("\\"):  # EN: Avoid breaking explicit line continuation (Python and some preprocessors).
            out_lines.append(raw_line)  # EN: Leave continuation lines unchanged.
            continue  # EN: Move to the next line.

        if style.marker in line_no_nl:  # EN: Avoid duplicating our generated comments on subsequent runs.
            out_lines.append(raw_line)  # EN: Keep already-annotated lines unchanged.
            continue  # EN: Move to the next line.

        comment = generate_comment(stripped=stripped, path=path)  # EN: Generate an English explanation for this line.
        updated_line = f"{line_no_nl}  {style.marker} {comment}{newline}"  # EN: Append a marker-tagged inline comment.
        out_lines.append(updated_line)  # EN: Store the updated line.
        changed = True  # EN: Mark that we changed at least one line.

    if not changed:  # EN: Return None to signal that the file already complied.
        return None  # EN: Caller can skip writing if unchanged.
    return "".join(out_lines)  # EN: Join all updated lines back into a single string.


def update_triple_quote_state(  # EN: Track whether we are inside a Python triple-quoted string.
    *,  # EN: Force keyword-only arguments for clarity.
    stripped: str,  # EN: A whitespace-trimmed line (no trailing newline).
    in_triple: bool,  # EN: Current “inside triple quotes” state.
    triple_delim: str | None,  # EN: Current triple-quote delimiter (''' or """), if inside.
) -> tuple[bool, str | None]:  # EN: Return the updated state (in_triple, delimiter).
    if in_triple:  # EN: If we are currently inside, look for the closing delimiter.
        if triple_delim and triple_delim in stripped:  # EN: Detect closing delimiter occurrence on this line.
            count = stripped.count(triple_delim)  # EN: Count occurrences to handle same-line open/close patterns.
            if count % 2 == 1:  # EN: Toggle only if the delimiter count is odd.
                return False, None  # EN: Exit triple-quote mode.
        return True, triple_delim  # EN: Stay in triple-quote mode.

    # EN: If we are not inside, detect a new triple-quoted string opening.
    for delim in ("'''", '"""'):  # EN: Check both possible triple-quote delimiters.
        if delim in stripped:  # EN: If the delimiter appears, toggle into triple-quote mode (if unmatched).
            count = stripped.count(delim)  # EN: Count occurrences to see whether it opens and closes on same line.
            if count % 2 == 1:  # EN: Only enter triple-quote mode when the delimiter is unmatched.
                return True, delim  # EN: Enter triple-quote mode with the chosen delimiter.
    return False, None  # EN: Remain outside triple quotes.


def is_comment_only_line(*, stripped: str, style: CommentStyle) -> bool:  # EN: Decide whether the line is a pure comment.
    if style is PY_STYLE:  # EN: Python comment-only lines start with "#".
        return stripped.startswith("#")  # EN: Return True if the line is a Python comment-only line.
    # EN: For C-like files, treat // and /* blocks as comment-only lines.
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("*/")  # EN: Detect common comment prefixes.


def shorten(text: str, max_len: int = 72) -> str:  # EN: Shorten long code fragments for readability in generated comments.
    compact = re.sub(r"\s+", " ", text).strip()  # EN: Collapse whitespace so we can safely truncate.
    if len(compact) <= max_len:  # EN: Return the string unchanged if it is already short enough.
        return compact  # EN: Provide the compact string as-is.
    return compact[: max_len - 1] + "…"  # EN: Truncate and add an ellipsis-like character.


def generate_comment(*, stripped: str, path: Path) -> str:  # EN: Produce a best-effort English comment for a given code line.
    if path.suffix == ".py":  # EN: Use Python-specific heuristics when the file is a Python module.
        return generate_python_comment(stripped)  # EN: Delegate to the Python comment generator.
    return generate_clike_comment(stripped)  # EN: Delegate to the C-like comment generator.


def generate_python_comment(stripped: str) -> str:  # EN: Heuristically describe a Python statement in English.
    if stripped.startswith("import "):  # EN: Handle plain import statements.
        return f"Import module(s): {shorten(stripped)}."  # EN: Describe the import at a high level.
    if stripped.startswith("from "):  # EN: Handle from-import statements.
        return f"Import symbol(s) from a module: {shorten(stripped)}."  # EN: Describe the from-import statement.
    if stripped.startswith("def "):  # EN: Handle function definitions.
        name = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)  # EN: Extract the function name when possible.
        fn = name[0] if name else "function"  # EN: Fall back if we cannot parse a name.
        return f"Define {fn} and its behavior."  # EN: Describe defining a function.
    if stripped.startswith("class "):  # EN: Handle class definitions.
        name = re.findall(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)  # EN: Extract the class name when possible.
        cls = name[0] if name else "class"  # EN: Fall back if we cannot parse a name.
        return f"Define {cls} and its methods/constants."  # EN: Describe defining a class.
    if stripped.startswith("return"):  # EN: Handle return statements.
        return f"Return a value: {shorten(stripped)}."  # EN: Describe returning from a function.
    if stripped.startswith("raise "):  # EN: Handle exception raising.
        return f"Raise an exception: {shorten(stripped)}."  # EN: Describe throwing an error.
    if stripped.startswith(("if ", "elif ")):  # EN: Handle conditional branches.
        return f"Branch on a condition: {shorten(stripped)}."  # EN: Describe conditional logic.
    if stripped == "else:":  # EN: Handle else blocks.
        return "Execute the fallback branch when prior conditions are false."  # EN: Describe the else branch purpose.
    if stripped.startswith("for "):  # EN: Handle for loops.
        return f"Iterate with a for-loop: {shorten(stripped)}."  # EN: Describe looping over an iterable.
    if stripped.startswith("while "):  # EN: Handle while loops.
        return f"Iterate while a condition holds: {shorten(stripped)}."  # EN: Describe a while loop.
    if stripped in {"break", "continue", "pass"}:  # EN: Handle simple control-flow statements.
        return f"Control flow statement: {stripped}."  # EN: Describe the control flow keyword.
    if stripped.startswith("try:"):  # EN: Handle try blocks.
        return "Start a try block for exception handling."  # EN: Explain the try block.
    if stripped.startswith("except "):  # EN: Handle except blocks.
        return f"Handle an exception case: {shorten(stripped)}."  # EN: Explain the exception handler.
    if stripped.startswith("with "):  # EN: Handle context managers.
        return f"Enter a context manager scope: {shorten(stripped)}."  # EN: Explain the with-statement.

    assign = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", stripped)  # EN: Try to parse simple assignments.
    if assign:  # EN: If this is an assignment, describe it more specifically.
        var = assign.group(1)  # EN: Capture the variable name.
        expr = assign.group(2)  # EN: Capture the assigned expression.
        return f"Assign {var} from expression: {shorten(expr)}."  # EN: Explain the assignment intent.

    aug = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*([+\-*/%]=)\s*(.+)$", stripped)  # EN: Try to parse augmented assignments.
    if aug:  # EN: If we recognized an augmented assignment, describe it.
        var = aug.group(1)  # EN: Capture the target variable.
        op = aug.group(2)  # EN: Capture the augmented operator.
        expr = aug.group(3)  # EN: Capture the right-hand expression.
        return f"Update {var} via {op} using: {shorten(expr)}."  # EN: Explain the in-place update.

    call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)  # EN: Detect a top-level function call.
    if call:  # EN: If this looks like a function call, name it.
        fn = call.group(1)  # EN: Extract the function name.
        if fn == "print":  # EN: Special-case print for clearer wording.
            return "Print formatted output to the console."  # EN: Explain printing behavior.
        return f"Call {fn}(...) to perform an operation."  # EN: Describe calling a function.

    return f"Execute statement: {shorten(stripped)}."  # EN: Fallback description when no pattern matches.


def generate_clike_comment(stripped: str) -> str:  # EN: Heuristically describe a C/C++/Java/JS/C# statement in English.
    if stripped.startswith("#include"):  # EN: Handle C/C++ includes.
        return f"Include a header dependency: {shorten(stripped)}."  # EN: Explain the include directive.
    if stripped.startswith("#define"):  # EN: Handle macros.
        return f"Define a preprocessor macro: {shorten(stripped)}."  # EN: Explain the macro definition.
    if stripped.startswith(("if ", "else if", "else", "switch")):  # EN: Handle conditional constructs.
        return f"Conditional control flow: {shorten(stripped)}."  # EN: Explain branching.
    if stripped.startswith(("for ", "while ", "do ")):  # EN: Handle loops.
        return f"Loop control flow: {shorten(stripped)}."  # EN: Explain iteration.
    if stripped.startswith("return"):  # EN: Handle return statements.
        return f"Return from the current function: {shorten(stripped)}."  # EN: Explain returning a value.
    if stripped in {"{", "}", "};"}:  # EN: Handle structural brace-only lines.
        return "Structure delimiter for a block or scope."  # EN: Explain braces at a high level.
    if stripped.endswith(";"):  # EN: Handle typical statement lines.
        return f"Execute a statement: {shorten(stripped)}."  # EN: Explain the statement generically.
    return f"Execute line: {shorten(stripped)}."  # EN: Fallback for anything else (e.g., signatures).


def generate_unit_docs(*, repo_root: Path, write: bool) -> Iterable[Path]:  # EN: Generate per-unit docs and yield doc file paths.
    implementations_root = repo_root / "docs" / "implementations"  # EN: Compute the base folder for generated unit docs.
    chapters = sorted(  # EN: Gather chapter folders like 01-*/02-*/ etc.
        [p for p in repo_root.iterdir() if p.is_dir() and re.match(r"^\d\d-", p.name)],  # EN: Filter by "NN-" prefix.
        key=lambda p: p.name,  # EN: Sort by folder name for stable output.
    )  # EN: Finish building the chapter list.

    for chapter_dir in chapters:  # EN: Iterate each chapter directory at the repo root.
        units = sorted(  # EN: Gather unit folders inside the chapter directory.
            [p for p in chapter_dir.iterdir() if p.is_dir() and re.match(r"^\d\d-", p.name)],  # EN: Filter by "NN-" prefix.
            key=lambda p: p.name,  # EN: Sort units by name for stable output.
        )  # EN: Finish building the unit list.

        for unit_dir in units:  # EN: Iterate each unit directory inside this chapter.
            unit_readme = unit_dir / "README.md"  # EN: Unit-level concept README path.
            if not unit_readme.exists():  # EN: Skip directories that do not represent a learning unit.
                continue  # EN: Move to the next candidate directory.

            source_files = sorted(  # EN: Collect all source files under this unit directory.
                [p for p in unit_dir.rglob("*") if p.is_file() and p.suffix in SOURCE_EXTENSIONS],  # EN: Filter by extensions.
                key=lambda p: str(p),  # EN: Sort by full relative path for deterministic ordering.
            )  # EN: Finish collecting source files.
            if not source_files:  # EN: Skip units with no source code files.
                continue  # EN: Move to the next unit.

            doc_path = implementations_root / chapter_dir.name / unit_dir.name / "README.md"  # EN: Compute doc output path.
            doc_path.parent.mkdir(parents=True, exist_ok=True)  # EN: Ensure the destination directory exists.

            doc_text = build_unit_doc(  # EN: Build the Markdown content for this unit.
                repo_root=repo_root,  # EN: Provide the repo root for relative paths.
                chapter_dir=chapter_dir,  # EN: Provide the chapter directory.
                unit_dir=unit_dir,  # EN: Provide the unit directory.
                unit_readme=unit_readme,  # EN: Provide the unit README path.
                source_files=source_files,  # EN: Provide the list of source files for this unit.
            )  # EN: Finish doc content generation.

            existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""  # EN: Load existing content if any.
            if existing != doc_text and write:  # EN: Only rewrite when content differs and writing is enabled.
                doc_path.write_text(doc_text, encoding="utf-8")  # EN: Write the updated doc content.
            yield doc_path  # EN: Yield the doc path so callers can report it.


def build_unit_doc(  # EN: Build a Traditional-Chinese per-unit implementation doc.
    *,  # EN: Force keyword-only arguments for clarity.
    repo_root: Path,  # EN: Repository root for relative paths.
    chapter_dir: Path,  # EN: The chapter directory.
    unit_dir: Path,  # EN: The unit directory.
    unit_readme: Path,  # EN: The unit README file.
    source_files: list[Path],  # EN: All source files found under this unit directory.
) -> str:  # EN: Return the generated Markdown content as a string.
    rel_unit = unit_dir.relative_to(repo_root)  # EN: Compute unit path relative to the repo root for display.
    rel_unit_readme = unit_readme.relative_to(repo_root)  # EN: Compute README path relative to the repo root.

    lines: list[str] = []  # EN: Collect Markdown lines for the output document.
    lines.append(f"# 實作說明：{unit_dir.name}（{chapter_dir.name}）\n")  # EN: Add a title line for the unit doc.

    lines.append("## 對應原始碼\n")  # EN: Start the “source mapping” section.
    lines.append(f"- 單元路徑：`{rel_unit}/`\n")  # EN: Record the unit folder path.
    lines.append(f"- 概念說明：`{rel_unit_readme}`\n")  # EN: Record the unit README path.
    lines.append("- 程式實作：\n")  # EN: Introduce the list of implementation files.
    for src in source_files:  # EN: List each source file as a bullet.
        lines.append(f"  - `{src.relative_to(repo_root)}`\n")  # EN: Add the relative path for the source file.

    lines.append("\n## 目標與背景\n")  # EN: Start the background section.
    lines.append("- 本文件補充該單元的「可執行方式」與「實作重點」，概念推導請先閱讀單元內 `README.md`。\n")  # EN: Explain how this doc relates to the unit README.
    lines.append("- 若同單元有 `*_manual` 與 `*_numpy` 兩版本，建議先跑 manual 版本理解流程，再用 library 版本對照結果。\n")  # EN: Provide guidance for manual vs library implementations.

    lines.append("\n## 如何執行\n")  # EN: Start the “how to run” section.
    lines.extend(build_run_instructions(repo_root=repo_root, unit_dir=unit_dir, source_files=source_files))  # EN: Add per-language commands.

    lines.append("\n## 核心做法（重點）\n")  # EN: Start the “core approach” section.
    lines.append("- 依照單元 `README.md` 的公式/定義，將步驟拆成可讀的函數（如向量加法、矩陣乘法、轉置等）。\n")  # EN: Describe the typical structure of implementations.
    lines.append("- 以小維度範例（2D/3D 或 2×2/3×3）輸出中間結果，方便驗算與理解。\n")  # EN: Mention the pedagogical approach used throughout the repo.

    snippet = extract_code_snippet(repo_root=repo_root, source_files=source_files)  # EN: Choose and extract a representative code snippet.
    if snippet:  # EN: Only add the snippet section when we successfully extracted something.
        language, rel_path_str, snippet_text = snippet  # EN: Unpack snippet metadata for rendering.
        lines.append("\n## 程式碼區段（節錄）\n")  # EN: Start the code excerpt section.
        lines.append(f"以下節錄自 `{rel_path_str}`（僅保留關鍵段落）：\n\n")  # EN: Introduce where the excerpt comes from.
        lines.append(f"```{language}\n{snippet_text}\n```\n")  # EN: Emit the fenced code block.

    lines.append("\n## 驗證方式\n")  # EN: Start the verification section.
    lines.append("- 直接執行同單元的不同語言/版本，確認計算結果一致（允許些微浮點誤差）。\n")  # EN: Describe cross-checking outputs.
    lines.append("- 若輸出包含矩陣/向量，請檢查維度是否符合定義（例如 `A(m×n)·x(n)` 應得到 `b(m)`）。\n")  # EN: Suggest dimension sanity checks.

    return "".join(lines)  # EN: Join all Markdown lines into a single string.


def build_run_instructions(*, repo_root: Path, unit_dir: Path, source_files: list[Path]) -> list[str]:  # EN: Build per-language run commands in Chinese.
    grouped: dict[str, list[Path]] = {}  # EN: Group source files by their language folder (python/cpp/etc.).
    for src in source_files:  # EN: Iterate each source file to group it.
        rel = src.relative_to(unit_dir)  # EN: Compute the unit-relative path for folder extraction.
        language_dir = rel.parts[0] if rel.parts else "unknown"  # EN: The first path part is typically the language folder.
        grouped.setdefault(language_dir, []).append(src)  # EN: Append this file to its language group.

    out: list[str] = []  # EN: Collect markdown lines for run instructions.
    for language_dir in sorted(grouped.keys()):  # EN: Emit sections in a stable order.
        files = grouped[language_dir]  # EN: Get the list of files for this language.
        header = language_dir.capitalize()  # EN: Create a human-readable heading label.
        out.append(f"### {header}\n")  # EN: Add a subheading for this language.

        # EN: Prefer extracting exact commands from file headers when available.
        extracted_cmds = extract_commands_from_headers(files=files)  # EN: Try to parse compile/run commands from comments.
        if extracted_cmds:  # EN: If we found commands, include them verbatim.
            out.append("以下指令多半可在檔頭註解中找到（依檔案可能略有不同）：\n\n")  # EN: Explain the provenance of commands.
            out.append("```bash\n")  # EN: Start a bash fenced block.
            out.append(f"cd {unit_dir.relative_to(repo_root)}/{language_dir}\n")  # EN: Always include the recommended working directory.
            for cmd in extracted_cmds:  # EN: List extracted commands.
                out.append(f"{cmd}\n")  # EN: Add each command line as-is.
            out.append("```\n")  # EN: Close the bash fenced block.
            continue  # EN: Skip fallback generation if we have extracted commands.

        # EN: Fallback commands by language and file extension.
        out.append("```bash\n")  # EN: Start a bash fenced block for fallback commands.
        out.append(f"cd {unit_dir.relative_to(repo_root)}/{language_dir}\n")  # EN: Change into the language folder.
        for src in sorted(files, key=lambda p: p.name):  # EN: Emit each file as a runnable option.
            out.append(f"{fallback_command_for_file(src)}\n")  # EN: Append a reasonable default command.
        out.append("```\n")  # EN: Close the bash fenced block.

    return out  # EN: Return the generated run instruction lines.


def extract_commands_from_headers(*, files: list[Path]) -> list[str]:  # EN: Extract compile/run commands from the top of each file.
    commands: list[str] = []  # EN: Accumulate extracted commands while keeping ordering.
    seen: set[str] = set()  # EN: Track uniqueness to avoid repeating identical commands.

    patterns = [  # EN: Define regex patterns for common command lines present in this repo.
        re.compile(r"\\bgcc\\b[^\\n]*"),  # EN: Match gcc compile commands.
        re.compile(r"\\bg\\+\\+\\b[^\\n]*"),  # EN: Match g++ compile commands.
        re.compile(r"\\bjavac\\b[^\\n]*"),  # EN: Match javac compile commands.
        re.compile(r"\\bjava\\b\\s+[^\\n]*"),  # EN: Match java run commands.
        re.compile(r"\\bnode\\b[^\\n]*"),  # EN: Match node run commands.
        re.compile(r"\\bdotnet\\b[^\\n]*"),  # EN: Match dotnet commands.
        re.compile(r"\\bcsc\\b[^\\n]*"),  # EN: Match csc (C# compiler) commands.
        re.compile(r"\\bpython\\b[^\\n]*"),  # EN: Match python run commands in headers.
    ]  # EN: Finish defining patterns.

    for file_path in files:  # EN: Check each file’s header comment for commands.
        header = read_header(file_path=file_path, max_lines=60)  # EN: Read a small prefix of the file.
        for line in header.splitlines():  # EN: Scan each header line.
            for pat in patterns:  # EN: Try each known command pattern.
                m = pat.search(line)  # EN: Search the line for a command.
                if not m:  # EN: Skip if no match was found.
                    continue  # EN: Try next pattern.
                cmd = m.group(0).strip()  # EN: Normalize whitespace at ends.
                if cmd in seen:  # EN: Avoid duplicates across multiple files.
                    continue  # EN: Skip repeated commands.
                seen.add(cmd)  # EN: Record the command as seen.
                commands.append(cmd)  # EN: Append the command preserving discovery order.

    return commands  # EN: Return the extracted command list (possibly empty).


def read_header(*, file_path: Path, max_lines: int) -> str:  # EN: Read the first N lines of a file as text.
    try:  # EN: Guard against encoding errors in unusual files.
        text = file_path.read_text(encoding="utf-8", errors="replace")  # EN: Read the file content as UTF-8 best-effort.
    except OSError:  # EN: Handle filesystem read errors gracefully.
        return ""  # EN: Return empty header on failure.
    return "\n".join(text.splitlines()[:max_lines])  # EN: Return only the requested number of lines.


def fallback_command_for_file(path: Path) -> str:  # EN: Provide a simple default run command when none is found in headers.
    name = path.name  # EN: Capture the file name for command construction.
    stem = path.stem  # EN: Capture the base name without extension.
    if path.suffix == ".py":  # EN: Python files run directly with python.
        return f"python {name}"  # EN: Return a python run command.
    if path.suffix == ".js":  # EN: JavaScript files run with node.
        return f"node {name}"  # EN: Return a node run command.
    if path.suffix == ".java":  # EN: Java files typically compile then run the class matching the filename.
        return f"javac {name} && java {stem}"  # EN: Return a compile-and-run command.
    if path.suffix == ".c":  # EN: C files compile with gcc; -lm is often needed for math usage in this repo.
        return f"gcc -std=c99 -O2 {name} -o {stem} -lm && ./{stem}"  # EN: Return a compile-and-run command.
    if path.suffix == ".cpp":  # EN: C++ files compile with g++ and C++17 in this repo.
        return f"g++ -std=c++17 -O2 {name} -o {stem} && ./{stem}"  # EN: Return a compile-and-run command.
    if path.suffix == ".cs":  # EN: C# standalone files can be compiled with csc if available.
        return f"csc {name} && ./{stem}.exe"  # EN: Return a compile-and-run command for C#.
    return f"# (no default command for {name})"  # EN: Provide a harmless placeholder for unknown types.


def extract_code_snippet(  # EN: Extract a small excerpt for docs.
    *,  # EN: Force keyword-only arguments for clarity.
    repo_root: Path,  # EN: Repository root so we can render relative paths in docs.
    source_files: list[Path],  # EN: Candidate source files for picking an excerpt.
) -> tuple[str, str, str] | None:  # EN: Return (fence_language, relative_path, snippet_text) when available.
    preferred = pick_preferred_source(source_files=source_files)  # EN: Pick a representative “primary” source file.
    if preferred is None:  # EN: Bail out if we cannot pick any file.
        return None  # EN: Signal no snippet available.

    language = fence_language_for_path(preferred)  # EN: Choose the fenced code block language tag.
    try:  # EN: Read the preferred file to extract a short snippet.
        text = preferred.read_text(encoding="utf-8", errors="replace")  # EN: Load source file content safely.
    except OSError:  # EN: Handle rare read errors.
        return None  # EN: Skip snippet if we cannot read.

    snippet = first_interesting_block(text=text, path=preferred, max_lines=30)  # EN: Extract a small “interesting” block of code.
    return language, str(preferred.relative_to(repo_root)), snippet  # EN: Return snippet metadata for doc rendering.


def pick_preferred_source(*, source_files: list[Path]) -> Path | None:  # EN: Pick the most representative file for excerpts.
    # EN: Prefer Python manual implementations to show core algorithms without heavy libraries.
    for p in source_files:  # EN: Scan in existing sorted order.
        if p.suffix == ".py" and "_manual" in p.name:  # EN: Detect manual Python files by name convention.
            return p  # EN: Return the first matching manual Python file.
    for p in source_files:  # EN: Next, prefer any Python file.
        if p.suffix == ".py":  # EN: Check for Python extension.
            return p  # EN: Return the first Python file.
    return source_files[0] if source_files else None  # EN: Fall back to the first file if nothing else matches.


def fence_language_for_path(path: Path) -> str:  # EN: Map file extensions to Markdown code fence language tags.
    return {  # EN: Return the language tag mapping.
        ".py": "python",  # EN: Python language tag.
        ".c": "c",  # EN: C language tag.
        ".cpp": "cpp",  # EN: C++ language tag.
        ".java": "java",  # EN: Java language tag.
        ".js": "javascript",  # EN: JavaScript language tag.
        ".cs": "csharp",  # EN: C# language tag.
    }.get(path.suffix, "text")  # EN: Default to text when unknown.


def first_interesting_block(*, text: str, path: Path, max_lines: int) -> str:  # EN: Extract a short code block suitable for docs.
    lines = text.splitlines()  # EN: Split source file into logical lines for scanning.

    # EN: For Python, try to skip the module docstring and jump to the first def/class.
    if path.suffix == ".py":  # EN: Apply Python-specific heuristics for better snippets.
        in_triple = False  # EN: Track whether we are inside a triple-quoted docstring.
        triple_delim: str | None = None  # EN: Track the delimiter that opened the docstring.

        for i, line in enumerate(lines):  # EN: Scan the file to find the first function/class definition.
            s = line.strip()  # EN: Normalize whitespace for matching.
            code_part = s.split("#", 1)[0].rstrip()  # EN: Ignore trailing comments when detecting triple-quote delimiters.
            if not s:  # EN: Skip blank lines.
                continue  # EN: Move to next line.
            if s.startswith("#"):  # EN: Skip comment-only lines.
                continue  # EN: Move to next line.

            # EN: Docstring skipping state machine (best-effort, not a full parser).
            if in_triple:  # EN: If currently inside docstring, look for the closing delimiter.
                if triple_delim and triple_delim in code_part and code_part.count(triple_delim) % 2 == 1:  # EN: Detect docstring close.
                    in_triple = False  # EN: Exit docstring mode.
                    triple_delim = None  # EN: Clear delimiter state.
                continue  # EN: Do not start snippets inside docstrings.
            if code_part.startswith(('"""', "'''")):  # EN: Detect start of a module docstring.
                triple_delim = '"""' if code_part.startswith('"""') else "'''"  # EN: Determine which delimiter is used.
                if code_part.count(triple_delim) % 2 == 1:  # EN: Enter docstring mode only if not closed on the same line.
                    in_triple = True  # EN: Mark that we are inside a docstring.
                continue  # EN: Skip the docstring opening line.

            if s.startswith(("def ", "class ")):  # EN: Prefer starting at function/class definitions.
                return "\n".join(lines[i : min(len(lines), i + max_lines)]).rstrip()  # EN: Return a def/class-centered snippet.

        # EN: If no def/class was found, fall back to the first non-comment line.
        for i, line in enumerate(lines):  # EN: Scan again to find a reasonable fallback anchor.
            s = line.strip()  # EN: Normalize whitespace for matching.
            if not s or s.startswith("#"):  # EN: Skip blanks and comment-only lines.
                continue  # EN: Move to next line.
            return "\n".join(lines[i : min(len(lines), i + max_lines)]).rstrip()  # EN: Return a fallback snippet.

    # EN: Non-Python fallback: start at the first non-comment, prefer function-like lines if possible.
    start = 0  # EN: Default snippet start index.
    for i, line in enumerate(lines):  # EN: Find the first “interesting” line.
        s = line.strip()  # EN: Trim whitespace for classification.
        if not s:  # EN: Skip blank lines.
            continue  # EN: Move to next line.
        if s.startswith(("//", "/*", "*", "*/")):  # EN: Skip comment-only lines.
            continue  # EN: Move to next line.
        if s.startswith(("int ", "double ", "public ", "function ")):  # EN: Prefer function/class starts in other languages.
            start = i  # EN: Set snippet start at this line.
            break  # EN: Stop scanning once we find a suitable anchor.
        start = i  # EN: Otherwise, keep the first non-comment line as a fallback anchor.
        break  # EN: Stop scanning once we have the first non-comment line.

    end = min(len(lines), start + max_lines)  # EN: Compute snippet end index with an upper bound.
    return "\n".join(lines[start:end]).rstrip()  # EN: Return the extracted block as a single string.


if __name__ == "__main__":  # EN: Allow running this script directly.
    raise SystemExit(main())  # EN: Exit with the return code from main().
