"""Two reproducible counts for the Python source in the code comparison."""

import ast
import io
import tokenize


def _docstrings(tree):
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return [
        node.body[0] for node in ast.walk(tree)
        if isinstance(node, owners) and ast.get_docstring(node, clean=False) is not None
    ]


def read_source(path):
    """Return (1-based line number, original text without newline) for each code line."""
    with tokenize.open(path) as stream:
        source = stream.read()
    lines = source.split("\n")
    # Python's AST reports UTF-8 byte offsets even for non-UTF-8 source files.
    code = [bytearray(line, "utf-8") for line in lines]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            start = len(lines[row - 1][:column].encode("utf-8"))
            del code[row - 1][start:]

    for doc in _docstrings(ast.parse(source, filename=str(path))):
        for row in range(doc.lineno, doc.end_lineno + 1):
            line = code[row - 1]
            start = doc.col_offset if row == doc.lineno else 0
            stop = doc.end_col_offset if row == doc.end_lineno else len(line)
            if row == doc.end_lineno and line[stop:].strip() == b";":
                stop = len(line)
            line[start:stop] = b" " * (stop - start)

    return [(row, lines[row - 1]) for row, line in enumerate(code, 1) if line.strip()]


def statement_count(path):
    """Count AST statements, including definitions and their bodies. Exclude docstrings."""
    with tokenize.open(path) as stream:
        tree = ast.parse(stream.read(), filename=str(path))
    return sum(isinstance(node, ast.stmt) for node in ast.walk(tree)) - len(_docstrings(tree))
