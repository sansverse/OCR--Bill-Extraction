import re
from typing import List, Dict


def parse_line_items(ocr_results: List[Dict]) -> List[Dict]:
    """
    Very simple heuristic parser:
    - Groups OCR boxes into rows by Y coordinate
    - Treats left text as item_name
    - Last numbers as qty, rate, amount
    """
    rows = _group_into_rows(ocr_results)
    rows = [r for r in rows if not _is_header_row(r)]

    items: List[Dict] = []
    for row in rows:
        item = _parse_row(row)
        if item and _should_include(item):
            items.append(item)

    return items

def _group_into_rows(ocr_results: List[Dict]) -> List[List[Dict]]:
    if not ocr_results:
        return []

    sorted_items = sorted(ocr_results, key=lambda x: x["bbox"][1])
    rows: List[List[Dict]] = []
    current: List[Dict] = []
    y_thresh = 25  # INCREASED from 18 to catch more rows
    
    for it in sorted_items:
        if not current:
            current.append(it)
            continue

        prev_y = current[-1]["bbox"][1]
        curr_y = it["bbox"][1]
        if abs(curr_y - prev_y) < y_thresh:
            current.append(it)
        else:
            if len(current) >= 2:
                rows.append(current)
                print(f"ROW FOUND: {len(current)} items")  # DEBUG
            current = [it]

    if len(current) >= 2:
        rows.append(current)

    print(f"TOTAL ROWS: {len(rows)}")  # DEBUG
    return rows

def _is_header_row(row: List[Dict]) -> bool:
    header_kw = {
        "item", "product", "description", "name",
        "qty", "quantity", "rate", "price", "amount", "total", "mrp"
    }
    text = " ".join(cell["text"].lower() for cell in row)
    matches = sum(1 for k in header_kw if k in text)
    return matches >= 2

def _parse_row(row: List[Dict]) -> Dict:
    row_sorted = sorted(row, key=lambda x: x["bbox"][0])
    texts = [c["text"].strip() for c in row_sorted]
    nums = []
    
    for t in texts:
        val = _extract_number(t)
        nums.append(val)

    numeric_indices = [i for i, v in enumerate(nums) if v is not None]
    
    # CHANGE: Accept rows with just 1 number (item name + amount)
    if len(numeric_indices) < 1:  # was < 2
        print(f"SKIP: Not enough numbers in {texts}")
        return None

    first_num_idx = numeric_indices[0]
    item_name = " ".join(texts[:first_num_idx]).strip()
    
    if not item_name or len(item_name) < 2:
        print(f"SKIP: Bad item name: {item_name}")
        return None

    numeric_values = [nums[i] for i in numeric_indices]

    # Smarter amount detection
    if len(numeric_values) >= 3:
        qty = numeric_values[-3]
        rate = numeric_values[-2]
        amt = numeric_values[-1]
    elif len(numeric_values) == 2:
        qty = numeric_values[-2]
        amt = numeric_values[-1]
        rate = amt / qty if qty > 0 else 0.0
    elif len(numeric_values) == 1:
        # Only one number: assume it's the amount
        qty = 1.0
        amt = numeric_values[0]
        rate = amt
    else:
        return None

    result = {
        "item_name": item_name,
        "item_quantity": float(qty),
        "item_rate": float(rate),
        "item_amount": float(amt),
    }
    
    print(f"PARSED: {result['item_name']} x{result['item_quantity']} @ {result['item_rate']} = {result['item_amount']}")
    return result


def _extract_number(text: str):
    cleaned = re.sub(r"[₹$€£,\s]", "", text)
    m = re.search(r"(\d+\.?\d*)", cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _should_include(item: Dict) -> bool:
    name = item["item_name"].lower()
    exclude = [
        "subtotal", "sub-total", "total", "grand total", "net total",
        "cgst", "sgst", "igst", "tax", "gst", "vat",
        "discount", "round off", "amount due"
    ]
    result = not any(k in name for k in exclude)
    if not result:
        print(f"EXCLUDED: {name}")  # DEBUG
    return result

