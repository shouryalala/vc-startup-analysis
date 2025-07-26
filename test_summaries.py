#!/usr/bin/env python3
"""Test summary generation on a subset of companies"""

import asyncio
import csv
from analyze_yc_company_parallel import add_summaries_and_scores

async def test_with_subset():
    """Test with first 3 companies only"""
    # Read CSV and take first 3
    companies = []
    with open("yc_company_analysis.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 3:
                companies.append(row)
            else:
                break
    
    # Write subset to test file
    test_file = "test_analysis.csv"
    fieldnames = list(companies[0].keys())
    
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(companies)
    
    print(f"Created test file with {len(companies)} companies")
    
    # Run summary generation on test file
    await add_summaries_and_scores(test_file)
    
    # Read and display results
    print("\nResults:")
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"\n{row['company_name']}:")
            print(f"  Summary: {row.get('one_line_summary', 'N/A')}")
            print(f"  Score: {row.get('company_score', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(test_with_subset())