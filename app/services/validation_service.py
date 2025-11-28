"""Simple validation helpers for parsed line items.

These are lightweight implementations intended to avoid import/runtime
errors while keeping behaviour reasonable for testing and development.
"""
from typing import List, Dict, Tuple
import statistics


def validate_line_item(item: Dict) -> Tuple[bool, float, List[str]]:
    """Validate a single parsed `item` dict.

    Returns (ok, confidence_estimate, issues_list).
    """
    issues = []
    try:
        qty = float(item.get("item_quantity", 0) or 0)
        amt = float(item.get("item_amount", 0) or 0)
    except Exception:
        return False, 0.0, ["invalid_numbers"]

    if qty <= 0:
        issues.append("non_positive_quantity")
    if amt <= 0:
        issues.append("non_positive_amount")

    # Very simple confidence heuristic: penalize missing fields
    conf = 0.95
    if not item.get("item_name"):
        conf -= 0.4
    if issues:
        conf -= 0.2 * len(issues)

    ok = len(issues) == 0
    conf = max(0.0, min(1.0, conf))
    return ok, conf, issues


def detect_outlier_amounts(items: List[Dict]) -> List[float]:
    """Return a list of amounts considered outliers using simple stats.

    If there are fewer than 2 items, return empty list.
    """
    amounts = []
    for it in items:
        try:
            amounts.append(float(it.get("item_amount", 0) or 0))
        except Exception:
            continue

    if len(amounts) < 2:
        return []

    mean = statistics.mean(amounts)
    stdev = statistics.pstdev(amounts)
    if stdev == 0:
        return []

    outliers = [a for a in amounts if abs(a - mean) > 3 * stdev]
    return outliers


def compute_overall_confidence(items: List[Dict]) -> float:
    """Compute a simple overall confidence value for a list of validated items.

    This implementation returns a value between 0.0 and 1.0. If there are no
    items, returns 0.0.
    """
    if not items:
        return 0.0

    # If items include a numeric `conf` field, average it; otherwise assume high
    # confidence for basic parsed items.
    confs = []
    for it in items:
        try:
            confs.append(float(it.get("conf", 0.95)))
        except Exception:
            confs.append(0.95)

    try:
        return float(sum(confs) / len(confs))
    except Exception:
        return 0.0
