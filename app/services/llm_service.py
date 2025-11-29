import json
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

# Try to import Groq, fallback gracefully if not installed
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Install with: pip install groq")

# Optionally load a local .env file if python-dotenv is installed. This is
# convenient for local development but we DO NOT store secrets in the repo.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not installed; that's fine — environment variables can be
    # provided by the OS or by your process manager.
    pass


def extract_items_with_llm(ocr_text: str) -> List[Dict]:
    """
    Use Groq LLM to parse OCR text and extract line items.
    Returns list of dicts with: item_name, item_quantity, item_rate, item_amount
    """
    if not GROQ_AVAILABLE:
        logger.warning("Groq not available, returning empty list")
        return []

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return []

    try:
        client = Groq(api_key=api_key)

        prompt = f"""You are an expert at parsing medical bills, invoices, pharmacy receipts, and financial documents.

Extract ALL line items from this OCR-extracted text. For each item, provide:
- item_name: The name/description of the item/service/drug (required, string)
- item_quantity: Quantity (number, default 1 if not specified)
- item_rate: Unit price/rate (number)
- item_amount: Total amount for this line (qty × rate, number)

IMPORTANT:
1. Include ALL items - don't skip any
2. Ignore header rows (like "S.No", "Item", "Description")
3. Ignore subtotals, totals, taxes, discounts
4. For each numeric value, treat it as: [description] [qty] x [rate] = [amount]
5. If only two numbers exist: first is qty, second is amount
6. Return ONLY valid JSON array, no markdown, no explanations

OCR TEXT:
{ocr_text}

Return JSON array only (no other text):"""

        logger.info("Calling Groq API...")
        message = client.messages.create(
            model="mixtral-8x7b-32768",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()
        logger.info(f"Groq response length: {len(response_text)} chars")

        # Try to parse JSON
        try:
            items = json.loads(response_text)
            if isinstance(items, list):
                logger.info(f"LLM extracted {len(items)} items")
                # Validate each item has required fields
                valid_items = []
                for item in items:
                    if all(k in item for k in ["item_name", "item_quantity", "item_rate", "item_amount"]):
                        valid_items.append({
                            "item_name": str(item["item_name"]).strip(),
                            "item_quantity": float(item["item_quantity"]),
                            "item_rate": float(item["item_rate"]),
                            "item_amount": float(item["item_amount"]),
                        })
                logger.info(f"Validated {len(valid_items)} items")
                return valid_items
            else:
                logger.error(f"LLM response is not a list: {type(items)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {str(e)[:100]}")
            logger.debug(f"Response was: {response_text[:200]}")
            return []

    except Exception as e:
        logger.error(f"LLM extraction failed: {str(e)}")
        return []
