"""Custom constraint management for synthetic data validation."""
import pandas as pd
from typing import List, Dict, Callable
import json


class CustomConstraint:
    """Define and validate a custom constraint on synthetic data."""
    
    def __init__(self, name: str, expression: str, tolerance: float = 0.01):
        """
        Initialize a constraint.
        
        Args:
            name: Constraint name (e.g., "w/c_ratio")
            expression: Python expression using column names (e.g., "abs(w_c - Water/Cement) < 0.01")
            tolerance: Acceptable tolerance for numerical constraints (default 0.01)
        """
        self.name = name
        self.expression = expression
        self.tolerance = tolerance
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply constraint to DataFrame, return rows that satisfy it."""
        try:
            # Create a safe namespace with column data and tolerance
            namespace = {col: df[col].values for col in df.columns}
            namespace['tolerance'] = self.tolerance
            namespace['abs'] = abs
            
            # Evaluate expression for each row
            mask = eval(self.expression, {"__builtins__": {}}, namespace)
            
            # If expression returns array-like, use it; otherwise assume all pass
            if isinstance(mask, (list, pd.Series)):
                return df[mask].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error validating constraint '{self.name}': {e}")
            return df


def interactive_add_constraints() -> List[CustomConstraint]:
    """Interactively prompt user to add custom constraints."""
    constraints = []
    print("\n=== Custom Constraint Editor ===")
    print("Define relationships between columns (e.g., 'w/c == Water / Cement')")
    print("Commands: 'add', 'list', 'remove', 'done'\n")
    
    while True:
        cmd = input("Enter command (add/list/remove/done): ").strip().lower()
        
        if cmd == 'done':
            break
        elif cmd == 'add':
            name = input("Constraint name (e.g., w_c_ratio): ").strip()
            if not name:
                print("Name cannot be empty")
                continue
            expr = input("Expression (e.g., abs(w_c - Water/Cement) < tolerance): ").strip()
            if not expr:
                print("Expression cannot be empty")
                continue
            try:
                tol = float(input("Tolerance (default 0.01): ").strip() or "0.01")
            except ValueError:
                tol = 0.01
            constraints.append(CustomConstraint(name, expr, tol))
            print(f"✓ Added constraint: {name}")
        
        elif cmd == 'list':
            if not constraints:
                print("No constraints defined yet")
            else:
                for i, c in enumerate(constraints):
                    print(f"{i}: {c.name} | {c.expression} (tolerance: {c.tolerance})")
        
        elif cmd == 'remove':
            if not constraints:
                print("No constraints to remove")
                continue
            try:
                idx = int(input("Constraint index to remove: ").strip())
                if 0 <= idx < len(constraints):
                    removed = constraints.pop(idx)
                    print(f"✓ Removed: {removed.name}")
                else:
                    print("Invalid index")
            except ValueError:
                print("Enter a valid index")
        
        else:
            print("Unknown command")
    
    return constraints


def apply_constraints(df: pd.DataFrame, constraints: List[CustomConstraint]) -> pd.DataFrame:
    """Apply all constraints to DataFrame, keeping only rows that satisfy all."""
    result = df.copy()
    for constraint in constraints:
        result = constraint.validate(result)
        removed = len(df) - len(result)
        if removed > 0:
            print(f"  Constraint '{constraint.name}': removed {removed} rows")
    return result


def save_constraints(constraints: List[CustomConstraint], path: str = 'custom_constraints.json') -> None:
    """Save constraints to JSON for reuse."""
    data = [{'name': c.name, 'expression': c.expression, 'tolerance': c.tolerance} for c in constraints]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Constraints saved to {path}")


def load_constraints(path: str = 'custom_constraints.json') -> List[CustomConstraint]:
    """Load constraints from JSON."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        constraints = [CustomConstraint(c['name'], c['expression'], c.get('tolerance', 0.01)) for c in data]
        print(f"Loaded {len(constraints)} constraints from {path}")
        return constraints
    except FileNotFoundError:
        print(f"No constraints file found at {path}")
        return []
