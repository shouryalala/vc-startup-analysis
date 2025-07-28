#!/usr/bin/env python3
"""
YC Company Analysis PDF Report Generator
Creates a comprehensive PDF report with 1 company per page
"""

import csv
import os
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Tuple
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.frames import Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
API_BASE_URL = "https://api.openai.com/v1"

if not API_KEY or not ASSISTANT_ID:
    raise ValueError("Please ensure OPENAI_API_KEY and ASSISTANT_ID are set in .env file")

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

# Prompt for summarizing criteria to 10 words
CRITERIA_SUMMARY_PROMPT = """
Based on the following analysis text, provide a concise evaluation in EXACTLY 10 words or less.
DO NOT return JSON, ratings, or any structured data.
DO NOT use ellipsis (...).
Just return a plain text summary.

Analysis text: {text}

Respond with ONLY a plain text summary (maximum 10 words). No JSON, no formatting, just text.
"""

# Prompt for generating highlights
HIGHLIGHTS_PROMPT = """
Based on the following company analysis, identify and describe the most rare, exceptional, or outlier qualities about this company and its founders. Focus on what makes them stand out from typical YC companies.

Company: {company_name}
Score: {company_score}/5
Analysis Data:
- Evidence of Early PMF: {evidence_of_early_pmf}
- Competitors: {incumbent_or_startup_alternatives_exist}
- Founder Experience: {founders_have_ft_work_experience}
- Top 10% Distinction: {founder_top_10_percent_distinction}
- Founder-Product Fit: {founders_right_fit_for_product}
- Problem Uniqueness: {problem_unique_and_specific}
- Expansion Opportunity: {opportunity_to_expand}
- TAM Size: {tam_size}

Provide exactly 3 bullet points highlighting the most exceptional or noteworthy aspects. Focus on rare signals like:
- Exceptional founder backgrounds (e.g., IOI medalist, Forbes 30u30)
- Unique technical achievements
- Unusual market insights
- Extraordinary traction metrics
- Distinctive competitive advantages

Format your response as follows:
• First highlight point
• Second highlight point
• Third highlight point

DO NOT return JSON or any structured data format. Use bullet points (•) only.
"""

# Prompt for generating verdict
VERDICT_PROMPT = """
Based on the company analysis and score, provide a final investment verdict.

Company: {company_name}
Score: {company_score}/5
One-line Summary: {one_line_summary}

Key Analysis Points:
- Early PMF: {evidence_of_early_pmf}
- Founder Distinction: {founder_top_10_percent_distinction}
- Problem Uniqueness: {problem_unique_and_specific}
- Market Size: {tam_size}

Write EXACTLY 2-4 sentences that:
1. Justify the score given
2. Highlight the key investment thesis (positive or negative)
3. Note any major risks or exceptional opportunities
4. Provide a clear recommendation

IMPORTANT: Return ONLY plain text sentences. 
DO NOT include any JSON, structured data, evaluation matrices, or formatting.
Just write 2-4 normal sentences as the verdict.
"""


class VerdictGenerator:
    """Generate verdicts using OpenAI Assistant"""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.assistant_id = ASSISTANT_ID
        self.base_url = API_BASE_URL
        self.headers = HEADERS
        
    async def create_thread(self) -> str:
        """Create a new conversation thread"""
        url = f"{self.base_url}/threads"
        async with self.session.post(url, headers=self.headers, json={}) as response:
            response.raise_for_status()
            data = await response.json()
            return data["id"]
    
    async def add_message(self, thread_id: str, content: str) -> None:
        """Add a message to the thread"""
        url = f"{self.base_url}/threads/{thread_id}/messages"
        async with self.session.post(
            url,
            headers=self.headers,
            json={"role": "user", "content": content}
        ) as response:
            response.raise_for_status()
    
    async def run_assistant(self, thread_id: str) -> str:
        """Run the assistant and get the response"""
        url = f"{self.base_url}/threads/{thread_id}/runs"
        
        # Start the run
        async with self.session.post(
            url,
            headers=self.headers,
            json={"assistant_id": self.assistant_id}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            run_id = data["id"]
        
        # Poll for completion
        max_attempts = 30
        for _ in range(max_attempts):
            await asyncio.sleep(2)
            
            async with self.session.get(
                f"{self.base_url}/threads/{thread_id}/runs/{run_id}",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                run_data = await response.json()
                
                if run_data["status"] == "completed":
                    break
                elif run_data["status"] in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Run failed: {run_data['status']}")
        
        # Get messages
        async with self.session.get(
            f"{self.base_url}/threads/{thread_id}/messages",
            headers=self.headers
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            for message in data["data"]:
                if message["role"] == "assistant":
                    return message["content"][0]["text"]["value"]
            
            raise Exception("No assistant response found")
    
    async def summarize_criteria(self, text: str) -> str:
        """Summarize criteria text to 10 words max"""
        prompt = CRITERIA_SUMMARY_PROMPT.format(text=text)
        
        thread_id = await self.create_thread()
        await self.add_message(thread_id, prompt)
        summary = await self.run_assistant(thread_id)
        
        # Clean up any JSON response
        summary = summary.strip()
        if summary.startswith('{') or summary.startswith('['):
            # If JSON was returned, extract first meaningful text
            try:
                import re
                # Find first quoted string that's not a key
                match = re.search(r'"[^"]+"\s*:\s*"([^"]+)"', summary)
                if match:
                    summary = match.group(1)
                else:
                    summary = "Evaluation pending"
            except:
                summary = "Evaluation pending"
        
        # Ensure max 10 words
        words = summary.split()
        if len(words) > 10:
            summary = ' '.join(words[:10])
        
        return summary
    
    async def generate_highlights(self, company_data: Dict[str, Any]) -> str:
        """Generate highlights about rare/exceptional qualities"""
        prompt = HIGHLIGHTS_PROMPT.format(
            company_name=company_data.get('company_name', 'Unknown'),
            company_score=company_data.get('company_score', '3'),
            evidence_of_early_pmf=company_data.get('evidence_of_early_pmf', 'N/A'),
            incumbent_or_startup_alternatives_exist=company_data.get('incumbent_or_startup_alternatives_exist', 'N/A'),
            founders_have_ft_work_experience=company_data.get('founders_have_ft_work_experience', 'N/A'),
            founder_top_10_percent_distinction=company_data.get('founder_top_10_percent_distinction', 'N/A'),
            founders_right_fit_for_product=company_data.get('founders_right_fit_for_product', 'N/A'),
            problem_unique_and_specific=company_data.get('problem_unique_and_specific', 'N/A'),
            opportunity_to_expand=company_data.get('opportunity_to_expand', 'N/A'),
            tam_size=company_data.get('tam_size', 'N/A')
        )
        
        thread_id = await self.create_thread()
        await self.add_message(thread_id, prompt)
        highlights = await self.run_assistant(thread_id)
        
        # Clean up any JSON response
        highlights = highlights.strip()
        if highlights.startswith('{') or highlights.startswith('['):
            # If JSON was returned, try to extract meaningful text
            try:
                import re
                # Look for any text after colons that's not a key
                matches = re.findall(r':\s*"([^"]+)"', highlights)
                if matches:
                    # Join the first few meaningful strings
                    highlights = ' '.join(matches[:3])
                else:
                    highlights = "This company shows strong potential based on the founder backgrounds and market opportunity."
            except:
                highlights = "This company shows strong potential based on the founder backgrounds and market opportunity."
        
        return highlights
    
    async def generate_verdict(self, company_data: Dict[str, Any]) -> str:
        """Generate a verdict for a company based on its analysis"""
        prompt = VERDICT_PROMPT.format(
            company_name=company_data.get('company_name', 'Unknown'),
            company_score=company_data.get('company_score', '3'),
            one_line_summary=company_data.get('one_line_summary', 'N/A'),
            evidence_of_early_pmf=company_data.get('evidence_of_early_pmf', 'N/A'),
            founder_top_10_percent_distinction=company_data.get('founder_top_10_percent_distinction', 'N/A'),
            problem_unique_and_specific=company_data.get('problem_unique_and_specific', 'N/A'),
            tam_size=company_data.get('tam_size', 'N/A')
        )
        
        thread_id = await self.create_thread()
        await self.add_message(thread_id, prompt)
        verdict = await self.run_assistant(thread_id)
        
        # Clean up verdict - remove any JSON that might appear
        verdict = verdict.strip()
        
        # If there's JSON at the end, cut it off
        json_start = verdict.find('{')
        if json_start > 0:  # Found JSON after some text
            verdict = verdict[:json_start].strip()
        
        # Also check for evaluation matrix pattern
        if 'EvaluationMatrix' in verdict or 'evaluationMatrix' in verdict:
            # Find where the JSON starts and cut it off
            for marker in ['{', '"EvaluationMatrix"', '"evaluationMatrix"']:
                pos = verdict.find(marker)
                if pos > 0:
                    verdict = verdict[:pos].strip()
                    break
        
        # Ensure we have at least some verdict
        if not verdict or len(verdict) < 20:
            verdict = f"With a score of {company_data.get('company_score', '3')}/5, this company shows potential but requires careful evaluation. The founding team and market opportunity are notable, though execution risks remain."
        
        return verdict


class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.create_custom_styles()
        
    def create_custom_styles(self):
        """Create custom paragraph styles"""
        # Company name style
        self.company_name_style = ParagraphStyle(
            'CompanyName',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        # Section header style
        self.section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#333333'),
            spaceBefore=8,
            spaceAfter=4,
            alignment=TA_LEFT
        )
        
        # Overview text style
        self.overview_style = ParagraphStyle(
            'Overview',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#444444'),
            alignment=TA_JUSTIFY,
            spaceAfter=8
        )
        
        # Verdict style
        self.verdict_style = ParagraphStyle(
            'Verdict',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1a1a1a'),
            fontName='Helvetica-Bold',
            alignment=TA_LEFT,
            spaceBefore=8
        )
    
    
    async def create_analysis_table(self, company_data: Dict[str, Any], summaries: Dict[str, str]) -> Table:
        """Create analysis summary table with summarized criteria"""
        # Define criteria and extract data (excluding "Based in SF")
        criteria_mapping = [
            ("Early PMF", "evidence_of_early_pmf"),
            ("Competitors", "incumbent_or_startup_alternatives_exist"),
            ("Founder Exp.", "founders_have_ft_work_experience"),
            ("Top 10%", "founder_top_10_percent_distinction"),
            ("Right Fit", "founders_right_fit_for_product"),
            ("Unique Problem", "problem_unique_and_specific"),
            ("Expansion", "opportunity_to_expand"),
            ("Large TAM", "tam_size")
        ]
        
        # Create table data
        table_data = [["Criteria", "Assessment"]]
        
        for label, key in criteria_mapping:
            value = company_data.get(key, "N/A")
            
            # Get pre-computed summary
            if value and value != "N/A":
                summary = summaries.get(f"{company_data['company_name']}_{key}", "No summary available")
                # Ensure it's wrapped in a Paragraph for proper text wrapping
                summary_para = Paragraph(summary, ParagraphStyle(
                    'TableCell',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    leading=11
                ))
            else:
                summary_para = Paragraph("No data available", ParagraphStyle(
                    'TableCell',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    leading=11
                ))
            
            table_data.append([label, summary_para])
        
        # Create table
        col_widths = [2*inch, 4.5*inch]
        table = Table(table_data, colWidths=col_widths)
        
        # Apply table style
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#333333')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafafa')]),
            
            # Padding
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        return table
    
    async def create_company_section(self, company_data: Dict[str, Any], summaries: Dict[str, str], highlights: str, verdict: str) -> List:
        """Create a complete company section"""
        elements = []
        
        # Company name
        company_name = company_data.get('company_name', 'Unknown Company')
        elements.append(Paragraph(company_name, self.company_name_style))
        
        # Horizontal line
        elements.append(Spacer(1, 2))
        elements.append(Table([['']], colWidths=[7*inch], rowHeights=[1],
                            style=TableStyle([('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc'))])))
        elements.append(Spacer(1, 8))
        
        # One-line summary
        one_line_summary = company_data.get('one_line_summary', 'No summary available')
        elements.append(Paragraph(one_line_summary, self.overview_style))
        elements.append(Spacer(1, 12))
        
        # Criteria Table
        elements.append(Paragraph("<b>Criteria Table</b>", self.section_header_style))
        table = await self.create_analysis_table(company_data, summaries)
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        # Highlights
        elements.append(Paragraph("<b>Highlights</b>", self.section_header_style))
        
        # Create bullet point style
        bullet_style = ParagraphStyle(
            'BulletPoint',
            parent=self.overview_style,
            leftIndent=20,
            bulletIndent=10
        )
        
        # Split highlights into bullet points
        bullet_points = []
        if '•' in highlights:
            # If already has bullet points, split by them
            points = highlights.split('•')
            for point in points:
                point = point.strip()
                if point:
                    bullet_points.append(point)
        else:
            # Fallback: treat as single paragraph
            bullet_points = [highlights]
        
        # Add each bullet point
        for point in bullet_points:
            elements.append(Paragraph(f"• {point}", bullet_style))
        
        elements.append(Spacer(1, 12))
        
        # Verdict with Score
        score = company_data.get('company_score', '3')
        elements.append(Paragraph(f"<b>Verdict: Score {score}/5</b>", self.section_header_style))
        elements.append(Paragraph(verdict, self.overview_style))
        
        return elements
    
    async def generate_pdf_report(self, csv_file: str, output_file: str = "yc_company_analysis_report.pdf"):
        """Generate the complete PDF report"""
        # Read CSV data
        companies = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            companies = list(reader)
        
        if not companies:
            print("No companies found in CSV file")
            return
        
        print(f"Generating PDF report for {len(companies)} companies...")
        
        # Initialize generator
        async with aiohttp.ClientSession() as session:
            generator = VerdictGenerator(session)
        
            # Page counter to skip footer on first page
            page_counter = {'page': 0}
            
            def add_footer(canvas, doc):
                """Add footer to pages (except first page)"""
                page_counter['page'] += 1
                if page_counter['page'] > 1:  # Skip footer on first page
                    canvas.saveState()
                    
                    # Draw horizontal line
                    canvas.setStrokeColor(colors.HexColor('#cccccc'))
                    canvas.setLineWidth(0.5)
                    canvas.line(0.5*inch, 0.75*inch, letter[0] - 0.5*inch, 0.75*inch)
                    
                    # Footer text
                    canvas.setFont('Helvetica', 9)
                    canvas.setFillColor(colors.HexColor('#666666'))
                    
                    # Left side - report title
                    canvas.drawString(0.75*inch, 0.5*inch, "S25 developer, fintech, and legal tech companies analysis")
                    
                    # Right side - author
                    canvas.drawRightString(letter[0] - 0.75*inch, 0.5*inch, "by Shourya Lala")
                    
                    canvas.restoreState()
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_file,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=1*inch  # Increased to make room for footer
            )
            
            # Build content
            story = []
            
            # Title page
            title_style = ParagraphStyle(
                'Title',
                parent=self.styles['Title'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                alignment=TA_CENTER,
                spaceAfter=20
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#333333'),
                alignment=TA_CENTER,
                spaceAfter=10
            )
            
            story.append(Spacer(1, 2*inch))
            story.append(Paragraph("S25 developer, fintech, and legal tech companies analysis", title_style))
            story.append(Paragraph("by Shourya Lala", subtitle_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Created: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
            story.append(Paragraph(f"Total Companies: {len(companies)}", self.styles['Normal']))
            story.append(PageBreak())
            
            # Step 1: Prepare all LLM tasks for parallel processing
            print("\nStep 1: Preparing LLM tasks for all companies...")
            criteria_tasks = []
            highlights_tasks = []
            verdict_tasks = []
            
            criteria_mapping = [
                ("evidence_of_early_pmf"),
                ("incumbent_or_startup_alternatives_exist"),
                ("founders_have_ft_work_experience"),
                ("founder_top_10_percent_distinction"),
                ("founders_right_fit_for_product"),
                ("problem_unique_and_specific"),
                ("opportunity_to_expand"),
                ("tam_size")
            ]
            
            for company in companies:
                company_name = company.get('company_name', 'Unknown')
                
                # Create tasks for criteria summaries
                for key in criteria_mapping:
                    value = company.get(key, "N/A")
                    if value and value != "N/A":
                        task_key = f"{company_name}_{key}"
                        criteria_tasks.append((task_key, generator.summarize_criteria(value)))
                
                # Create task for highlights
                highlights_tasks.append((company_name, generator.generate_highlights(company)))
                
                # Create task for verdict
                verdict_tasks.append((company_name, generator.generate_verdict(company)))
            
            # Step 2: Run all LLM tasks in parallel
            print(f"\nStep 2: Running {len(criteria_tasks) + len(highlights_tasks) + len(verdict_tasks)} LLM calls in parallel...")
            
            # Process criteria summaries
            summaries = {}
            if criteria_tasks:
                print(f"  - Processing {len(criteria_tasks)} criteria summaries...")
                criteria_results = await asyncio.gather(*[task for _, task in criteria_tasks], return_exceptions=True)
                for i, (key, _) in enumerate(criteria_tasks):
                    if not isinstance(criteria_results[i], Exception):
                        summaries[key] = criteria_results[i]
                    else:
                        summaries[key] = "Error generating summary"
            
            # Process highlights
            highlights_dict = {}
            if highlights_tasks:
                print(f"  - Processing {len(highlights_tasks)} highlights...")
                highlights_results = await asyncio.gather(*[task for _, task in highlights_tasks], return_exceptions=True)
                for i, (company_name, _) in enumerate(highlights_tasks):
                    if not isinstance(highlights_results[i], Exception):
                        highlights_dict[company_name] = highlights_results[i]
                    else:
                        highlights_dict[company_name] = "Error generating highlights"
            
            # Process verdicts
            verdict_dict = {}
            if verdict_tasks:
                print(f"  - Processing {len(verdict_tasks)} verdicts...")
                verdict_results = await asyncio.gather(*[task for _, task in verdict_tasks], return_exceptions=True)
                for i, (company_name, _) in enumerate(verdict_tasks):
                    if not isinstance(verdict_results[i], Exception):
                        verdict_dict[company_name] = verdict_results[i]
                    else:
                        verdict_dict[company_name] = "Error generating verdict"
            
            # Step 3: Generate PDF pages with pre-computed results
            print("\nStep 3: Generating PDF pages...")
            for i, company in enumerate(companies):
                company_name = company.get('company_name', 'Unknown')
                print(f"  - Creating page for {company_name}...")
                
                # Get pre-computed results
                company_highlights = highlights_dict.get(company_name, "No highlights available")
                company_verdict = verdict_dict.get(company_name, "No verdict available")
                
                # Create company section with pre-computed results
                company_elements = await self.create_company_section(
                    company, 
                    summaries, 
                    company_highlights, 
                    company_verdict
                )
                
                # Add to story
                story.extend(company_elements)
                
                # Page break after each company (except last)
                if i < len(companies) - 1:
                    story.append(PageBreak())
            
            # Build PDF with footer on each page
            doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
            print(f"\nPDF report generated successfully: {output_file}")


async def main():
    """Main function"""
    csv_file = "yc_company_analysis.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run the analysis first.")
        return
    
    generator = PDFReportGenerator()
    await generator.generate_pdf_report(csv_file)


if __name__ == "__main__":
    asyncio.run(main())