import json
from typing import Dict, Any
import pandas as pd


def generate_metadata(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a simple column-type mapping for the dataset.

    Preference order:
    1. If synthpop's MissingDataHandler is available, use its dtype detection.
    2. Otherwise fall back to pandas-based heuristic.

    Returns a mapping: {column_name: 'numerical'|'categorical'}
    """
    try:
        from synthpop import MissingDataHandler
        md = MissingDataHandler()
        metadata = md.get_column_dtypes(df)
        # md.get_column_dtypes returns a mapping for synthpop; normalize values
        normalized = {}
        for col, v in metadata.items():
            typ = str(v).lower()
            if 'num' in typ or 'int' in typ or 'float' in typ:
                normalized[col] = 'numerical'
            else:
                normalized[col] = 'categorical'
        return normalized
    except Exception:
        # Fallback: pandas heuristics
        mapping = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                mapping[col] = 'numerical'
            else:
                mapping[col] = 'categorical'
        return mapping


def interactive_edit_metadata(metadata: Dict[str, str]) -> Dict[str, str]:
    """Interactive CLI to edit metadata by index.

    Commands:
    - Enter an index number followed by a space and `n` or `c`, e.g. `3 n` to make column 3 numerical.
    - Enter `done` to finish and return the edited metadata.
    - Enter `show` to reprint the columns.
    """
    cols = list(metadata.keys())
    def print_cols():
        print("Current columns and types:")
        for i, c in enumerate(cols):
            print(f"{i}: {c} -> {metadata[c]}")

    print_cols()
    while True:
        try:
            s = input("Edit (index n/c), 'show', or 'done': ").strip()
        except EOFError:
            print("Input closed; finishing edits.")
            break
        if not s:
            continue
        if s.lower() == 'done':
            break
        if s.lower() == 'show':
            print_cols()
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                idx = int(parts[0])
                if idx < 0 or idx >= len(cols):
                    print("Index out of range")
                    continue
            except ValueError:
                print("First token must be an integer index or 'show'/'done'")
                continue
            cmd = parts[1].lower()
            if cmd == 'n':
                metadata[cols[idx]] = 'numerical'
            elif cmd == 'c':
                metadata[cols[idx]] = 'categorical'
            else:
                print("Unknown type command; use 'n' or 'c'")
            print(f"Set {cols[idx]} -> {metadata[cols[idx]]}")
        else:
            print("Invalid command format; try '3 n' or 'show' or 'done'")
    return metadata


def convert_to_sdv_metadata(metadata: Dict[str, str], table_name: str = 'table') -> Dict[str, Any]:
    """Convert the simple metadata mapping to an SDV single-table metadata dict similar to sdv_metadata.json in this repo."""
    columns = {}
    for col, t in metadata.items():
        sdtype = 'numerical' if t == 'numerical' else 'categorical'
        columns[col] = {'sdtype': sdtype}
    sdv_meta = {
        'tables': {
            table_name: {'columns': columns}
        },
        'relationships': [],
        'METADATA_SPEC_VERSION': 'V1'
    }
    return sdv_meta


def save_as_sdv_json(metadata: Dict[str, str], path: str = 'sdv_metadata.json', table_name: str = 'table') -> None:
    sdv_meta = convert_to_sdv_metadata(metadata, table_name=table_name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sdv_meta, f, ensure_ascii=False, indent=4)


def load_sdv_json(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        sd = json.load(f)
    # Attempt to extract first table columns
    tables = sd.get('tables', {})
    if not tables:
        return {}
    first_table = next(iter(tables.keys()))
    cols = tables[first_table].get('columns', {})
    mapping = {}
    for col, props in cols.items():
        sdtype = props.get('sdtype', 'categorical')
        if sdtype == 'numerical':
            mapping[col] = 'numerical'
        else:
            mapping[col] = 'categorical'
    return mapping
