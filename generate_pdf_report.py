#!/usr/bin/env python3
"""
YC Company Analysis PDF Report Generator
Creates a compact PDF report with 2 companies per page
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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
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
    
    async def generate_verdict(self, company_data: Dict[str, Any]) -> str:
        """Generate a verdict for a company based on its analysis"""
        prompt = f"""
Based on the following YC company analysis, provide a concise 1-2 sentence overall verdict that synthesizes all the key findings. Focus on investment potential and key strengths/weaknesses.

Company: {company_data.get('company_name', 'Unknown')}

Analysis:
- Early PMF: {company_data.get('evidence_of_early_pmf', 'N/A')}
- Competitors: {company_data.get('incumbent_or_startup_alternatives_exist', 'N/A')}
- Founder Experience: {company_data.get('founders_have_ft_work_experience', 'N/A')}
- Top 10% Distinction: {company_data.get('founder_top_10_percent_distinction', 'N/A')}
- Founder-Product Fit: {company_data.get('founders_right_fit_for_product', 'N/A')}
- Problem Uniqueness: {company_data.get('problem_unique_and_specific', 'N/A')}
- Expansion Opportunity: {company_data.get('opportunity_to_expand', 'N/A')}
- TAM: {company_data.get('tam_size', 'N/A')}

Provide ONLY the 1-2 sentence verdict. No additional text or explanation.
"""
        
        thread_id = await self.create_thread()
        await self.add_message(thread_id, prompt)
        verdict = await self.run_assistant(thread_id)
        return verdict.strip()


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
    
    def extract_company_description(self, launch_content: str) -> str:
        """Extract a brief description from launch post content"""
        if not launch_content:
            return "No description available."
        
        # Take first 2-3 sentences or ~200 characters
        sentences = launch_content.split('.')[:3]
        description = '. '.join(sentences).strip()
        
        # Clean up common patterns
        description = description.replace('\n', ' ')
        description = ' '.join(description.split())  # Remove extra spaces
        
        # Limit length
        if len(description) > 250:
            description = description[:247] + "..."
        elif not description.endswith('.'):
            description += '.'
            
        return description
    
    def create_analysis_table(self, company_data: Dict[str, Any]) -> Table:
        """Create a compact analysis summary table"""
        # Define criteria and extract data
        criteria_mapping = [
            ("Early PMF", "evidence_of_early_pmf"),
            ("Competitors", "incumbent_or_startup_alternatives_exist"),
            ("Founder Exp.", "founders_have_ft_work_experience"),
            ("Top 10%", "founder_top_10_percent_distinction"),
            ("Right Fit", "founders_right_fit_for_product"),
            ("Unique Problem", "problem_unique_and_specific"),
            ("Expansion", "opportunity_to_expand"),
            ("Large TAM", "tam_size"),
            ("Based in SF", "founders_based_in_sf")
        ]
        
        # Create table data
        table_data = [["Criteria", "Assessment"]]
        
        for label, key in criteria_mapping:
            value = company_data.get(key, "N/A")
            
            # Determine checkmark or X based on content
            if isinstance(value, str):
                value_lower = value.lower()
                if any(word in value_lower for word in ["yes", "strong", "excellent", "significant", "substantial", "large"]):
                    symbol = "✓"
                    color = colors.green
                elif any(word in value_lower for word in ["no", "error", "failed", "none", "limited"]):
                    symbol = "✗"
                    color = colors.red
                else:
                    symbol = "~"
                    color = colors.orange
            else:
                symbol = "?"
                color = colors.grey
            
            # Create brief summary (max 6-7 words)
            if value and value != "N/A":
                words = value.split()[:6]
                brief_summary = ' '.join(words)
                if len(words) < len(value.split()):
                    brief_summary += "..."
            else:
                brief_summary = "No data"
            
            # Add colored symbol
            table_data.append([
                label,
                f"{symbol} {brief_summary}"
            ])
        
        # Create table
        col_widths = [2.5*inch, 3.5*inch]
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
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        return table
    
    def create_company_section(self, company_data: Dict[str, Any], description: str, verdict: str) -> List:
        """Create a complete company section"""
        elements = []
        
        # Company name
        elements.append(Paragraph(company_data.get('company_name', 'Unknown Company'), self.company_name_style))
        
        # Horizontal line
        elements.append(Spacer(1, 2))
        elements.append(Table([['']], colWidths=[7*inch], rowHeights=[1],
                            style=TableStyle([('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#cccccc'))])))
        elements.append(Spacer(1, 8))
        
        # Overview section
        elements.append(Paragraph("<b>Overview</b>", self.section_header_style))
        elements.append(Paragraph(description, self.overview_style))
        
        # Analysis table
        elements.append(Paragraph("<b>Analysis Summary</b>", self.section_header_style))
        elements.append(self.create_analysis_table(company_data))
        
        # Verdict
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("<b>Verdict:</b> " + verdict, self.verdict_style))
        
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
        
        # Generate verdicts for all companies
        verdicts = {}
        async with aiohttp.ClientSession() as session:
            generator = VerdictGenerator(session)
            
            print("Generating verdicts...")
            for i, company in enumerate(companies):
                company_name = company.get('company_name', 'Unknown')
                print(f"  {i+1}/{len(companies)}: {company_name}")
                
                try:
                    verdict = await generator.generate_verdict(company)
                    verdicts[company_name] = verdict
                except Exception as e:
                    print(f"    Error generating verdict: {e}")
                    verdicts[company_name] = "Unable to generate verdict due to processing error."
        
        # Create PDF
        doc = SimpleDocTemplate(
            output_file,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
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
        
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("YC Company Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
        story.append(Paragraph(f"Total Companies: {len(companies)}", self.styles['Normal']))
        story.append(PageBreak())
        
        # Process companies in pairs
        for i in range(0, len(companies), 2):
            # First company
            company1 = companies[i]
            company1_name = company1.get('company_name', 'Unknown')
            
            # Extract description from launch_post_content if available
            launch_content = ""
            # Try to find the company in the input file to get launch_post_content
            try:
                with open('company_input.json', 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                    for comp in input_data:
                        if comp.get('company_name') == company1_name:
                            launch_content = comp.get('launch_post_content', '')
                            break
            except:
                pass
            
            description1 = self.extract_company_description(launch_content) if launch_content else \
                          f"{company1_name} is a YC company focused on innovative solutions."
            verdict1 = verdicts.get(company1_name, "Analysis pending.")
            
            # Create first company section
            company1_elements = self.create_company_section(company1, description1, verdict1)
            
            # Keep first company together
            story.append(KeepTogether(company1_elements))
            
            # Add separator between companies on same page
            if i + 1 < len(companies):
                story.append(Spacer(1, 0.3*inch))
                story.append(Table([['']], colWidths=[7*inch], rowHeights=[2],
                                 style=TableStyle([('LINEABOVE', (0, 0), (-1, 0), 2, colors.HexColor('#666666'))])))
                story.append(Spacer(1, 0.3*inch))
                
                # Second company
                company2 = companies[i + 1]
                company2_name = company2.get('company_name', 'Unknown')
                
                # Extract description
                launch_content2 = ""
                try:
                    with open('company_input.json', 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                        for comp in input_data:
                            if comp.get('company_name') == company2_name:
                                launch_content2 = comp.get('launch_post_content', '')
                                break
                except:
                    pass
                
                description2 = self.extract_company_description(launch_content2) if launch_content2 else \
                              f"{company2_name} is a YC company focused on innovative solutions."
                verdict2 = verdicts.get(company2_name, "Analysis pending.")
                
                # Create second company section
                company2_elements = self.create_company_section(company2, description2, verdict2)
                
                # Keep second company together
                story.append(KeepTogether(company2_elements))
            
            # Page break after each pair
            if i + 2 < len(companies):
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
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