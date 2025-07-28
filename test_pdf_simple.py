#!/usr/bin/env python3
"""Test PDF generation with simplified API calls"""

import asyncio
import csv
from generate_pdf_report import PDFReportGenerator

async def test_pdf():
    """Test PDF generation with first company"""
    # Read first company from CSV
    with open("yc_company_analysis.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        company = next(reader)  # Get first company
    
    print(f"Testing with {company['company_name']}...")
    
    # Create generator and generate PDF
    generator = PDFReportGenerator()
    await generator.generate_pdf_report("yc_company_analysis.csv", "test_report.pdf")
    
    print("Done! Check test_report.pdf")

if __name__ == "__main__":
    asyncio.run(test_pdf())