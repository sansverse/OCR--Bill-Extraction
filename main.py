from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any
import requests
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import os

from app.models import (
    ExtractionResponse,
    BillExtractionData,
    BillPage,
    LineItem,
    QualityMetrics,
)
from app.services.ocr_service import extract_text_with_easyocr
from app.services.llm_service import extract_items_with_llm
from app.services.extraction_service import parse_line_items
from app.services.validation_service import (
    validate_line_item,
    detect_outlier_amounts,
    compute_overall_confidence,
)
from utils.image_preprocessing import (
    deskew_image,
    enhance_contrast,
    image_quality_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bill Extraction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(body: Dict[str, Any]):
    try:
        url = body.get("document")
        if not url:
            raise HTTPException(status_code=400, detail="Missing 'document' field")

        logger.info(f"Processing: {url[:100]}...")
        
        # Download image
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Preprocess
        logger.info("Preprocessing image...")
        img_cv = deskew_image(img_cv)
        img_cv = enhance_contrast(img_cv)
        qual = image_quality_score(img_cv)
        logger.info(f"Image quality score: {qual:.3f}")

        # OCR
        logger.info("Running EasyOCR...")
        ocr_results = extract_text_with_easyocr(img_cv)
        logger.info(f"OCR found {len(ocr_results)} text regions")

        # Convert OCR results to readable text for LLM
        ocr_text = "\n".join([item["text"] for item in ocr_results])
        logger.info(f"OCR text length: {len(ocr_text)} characters")

        # Try LLM extraction first
        logger.info("Attempting LLM-based extraction...")
        raw_items = extract_items_with_llm(ocr_text)
        llm_used = bool(raw_items)

        # Fallback to regex if LLM fails or returns nothing
        if not llm_used:
            logger.warning("LLM extraction returned no items, falling back to regex parsing...")
            raw_items = parse_line_items(ocr_results)
            llm_used = False

        logger.info(f"Extracted {len(raw_items)} raw items (llm_used={llm_used})")

        # Validate items
        valid_items = []
        low_conf_names = []
        suspicious = []

        for it in raw_items:
            ok, conf, issues = validate_line_item(it)
            if ok:
                if conf < 0.85:
                    low_conf_names.append(it["item_name"])
                valid_items.append(it)
            else:
                suspicious.append({"item": it, "issues": issues})

        logger.info(f"Valid items after validation: {len(valid_items)}")
        logger.info(f"Suspicious items: {len(suspicious)}")

        # Calculate metrics
        reconciled_amount = sum(i["item_amount"] for i in valid_items)
        outliers = detect_outlier_amounts(valid_items)
        overall_conf = compute_overall_confidence(valid_items)

        logger.info(f"Reconciled amount: ₹{reconciled_amount:.2f}")
        logger.info(f"Outliers detected: {len(outliers)}")

        # Build response
        line_items = [LineItem(**i) for i in valid_items]
        page = BillPage(page_no="1", bill_items=line_items)
        bill_data = BillExtractionData(
            pagewise_line_items=[page],
            total_item_count=len(valid_items),
            reconciled_amount=round(reconciled_amount, 2),
        )

        qm = QualityMetrics(
            image_quality_score=qual,
            extraction_confidence=overall_conf,
            items_with_low_confidence=low_conf_names,
            outliers_detected=[f"{a:.2f}" for a in outliers],
            flags=[],
        )

        return ExtractionResponse(
            is_success=True,
            data=bill_data.model_dump(),
            quality_metrics=qm.model_dump(),
            reconciliation_report={
                "suspicious_items": len(suspicious),
                "valid_items": len(valid_items),
                "llm_used": llm_used,
            },
        )

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
