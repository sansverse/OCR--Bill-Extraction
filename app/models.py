from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class LineItem(BaseModel):
    item_name: str
    item_quantity: float
    item_rate: float
    item_amount: float


class BillPage(BaseModel):
    page_no: str
    bill_items: List[LineItem]


class BillExtractionData(BaseModel):
    pagewise_line_items: List[BillPage]
    total_item_count: int
    reconciled_amount: float


class QualityMetrics(BaseModel):
    image_quality_score: float
    extraction_confidence: float
    items_with_low_confidence: List[str]
    outliers_detected: List[str]
    flags: List[str]


class ExtractionResponse(BaseModel):
    is_success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    quality_metrics: Optional[QualityMetrics] = None
    reconciliation_report: Optional[Dict[str, Any]] = None
