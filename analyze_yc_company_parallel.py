#!/usr/bin/env python3
"""
YC Company Analysis Script - Parallel Version
Analyzes multiple YC companies concurrently using OpenAI Assistant API
"""

import json
import sys
import asyncio
import csv
import os
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
API_BASE_URL = "https://api.openai.com/v1"
MODEL = "o1-preview"  # Using o1-preview as o3-pro is not available

# Validate environment variables
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
if not ASSISTANT_ID:
    raise ValueError("ASSISTANT_ID not found in environment variables. Please create a .env file with your assistant ID.")

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

# Summary and scoring prompt template
SUMMARY_AND_SCORE_PROMPT = """
Based on the comprehensive analysis below, provide:
1. A one-line summary (crisp, blunt, under 15 words)
2. A score from 1-5

Company Analysis Data:
- Company Name: {company_name}
- Evidence of Early PMF: {evidence_of_early_pmf}
- Competitors Exist: {incumbent_or_startup_alternatives_exist}
- Founders Have FT Experience: {founders_have_ft_work_experience}
- Top 10% Distinction: {founder_top_10_percent_distinction}
- Founders Right Fit: {founders_right_fit_for_product}
- Problem Unique: {problem_unique_and_specific}
- Expansion Opportunity: {opportunity_to_expand}
- TAM Size: {tam_size}
- Direct Competitors: {direct_competitors}
- Based in SF: {founders_based_in_sf}

One-line Summary Guidelines:
- Be direct and blunt about what they're building
- Format: "Building [what] for [who] with [unique aspect]"
- Highlight anything special or refreshing
- Maximum 15 words

Scoring Rubric (1-5):
- 5 (Exceptional): Top 2% - Truly exceptional company with brilliant founders and massive opportunity
- 4 (Excellent): Top 15% - Brilliant founders perfectly suited for problem, strong on all criteria
- 3 (Average): Middle 60% - Solid company with typical mix of strengths and weaknesses
- 2 (Below Average): Bottom 20% - Significant challenges, missing key elements
- 1 (Poor): Bottom 3% - Major red flags across multiple dimensions

Most companies should score 3. Reserve 4 for truly excellent teams. 5 should be extremely rare.

Respond with ONLY a JSON object in this format:
{{
    "one_line_summary": "your summary here",
    "company_score": 3
}}
"""

# Analysis prompt template
ANALYSIS_PROMPT = """
Analyze the following YC company data and provide a comprehensive assessment:

Company Data:
{company_data}

Please analyze and provide insights on:
1. How specific and unique is the idea? Does it have any unique insight attached to it?
2. How many customers have the founders spoken to? (infer from the launch post and context)
3. Are the founders the right fit for building this product?
4. Evidence of early product-market fit?
5. Do incumbent or startup alternatives exist?
6. Do founders have full-time work experience?
7. Does any founder's experience/background distinguish them as top 10% within their batch?
8. Is the problem they're solving unique and specific enough to allow them to win?
9. Is there opportunity to expand? What's the TAM?
10. Who are the direct competitors?
11. Are the founders based in SF?

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any text before or after the JSON. Do not wrap it in markdown code blocks.

Provide your response as a JSON object with these exact keys:
{{
    "evidence_of_early_pmf": "description here",
    "incumbent_or_startup_alternatives_exist": "description here",
    "founders_have_ft_work_experience": "description here",
    "founder_top_10_percent_distinction": "description here",
    "founders_right_fit_for_product": "description here",
    "problem_unique_and_specific": "description here",
    "opportunity_to_expand": "description here",
    "tam_size": "description here",
    "direct_competitors": "description here",
    "founders_based_in_sf": "description here"
}}
"""


class YCCompanyAnalyzer:
    def __init__(self, session: aiohttp.ClientSession, progress_callback=None):
        self.session = session
        self.api_key = API_KEY
        self.assistant_id = ASSISTANT_ID
        self.base_url = API_BASE_URL
        self.headers = HEADERS
        self.max_polling_attempts = 60
        self.polling_interval = 2
        self.progress_callback = progress_callback

    async def create_thread(self) -> str:
        """Create a new conversation thread"""
        url = f"{self.base_url}/threads"
        
        try:
            async with self.session.post(url, headers=self.headers, json={}) as response:
                response.raise_for_status()
                data = await response.json()
                return data["id"]
        except aiohttp.ClientError as e:
            print(f"Error creating thread: {e}")
            raise

    async def add_message(self, thread_id: str, content: str) -> None:
        """Add a message to the thread"""
        url = f"{self.base_url}/threads/{thread_id}/messages"
        
        try:
            async with self.session.post(
                url,
                headers=self.headers,
                json={
                    "role": "user",
                    "content": content
                }
            ) as response:
                response.raise_for_status()
        except aiohttp.ClientError as e:
            print(f"Error adding message: {e}")
            raise

    async def run_assistant(self, thread_id: str) -> str:
        """Run the assistant and get the response"""
        url = f"{self.base_url}/threads/{thread_id}/runs"
        
        # Start the run
        try:
            async with self.session.post(
                url,
                headers=self.headers,
                json={
                    "assistant_id": self.assistant_id
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error response: {response.status}")
                    print(f"Error details: {error_text}")
                    response.raise_for_status()
                    
                data = await response.json()
                run_id = data["id"]
        except aiohttp.ClientError as e:
            print(f"Error starting run: {e}")
            raise

        # Poll for completion
        attempts = 0
        while attempts < self.max_polling_attempts:
            await asyncio.sleep(self.polling_interval)
            
            # Check run status
            try:
                async with self.session.get(
                    f"{self.base_url}/threads/{thread_id}/runs/{run_id}",
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    run_data = await response.json()
                    
                    status = run_data["status"]
                    
                    if status == "completed":
                        break
                    elif status in ["failed", "cancelled", "expired"]:
                        error_msg = f"Run failed with status: {status}"
                        if "last_error" in run_data and run_data["last_error"]:
                            error_msg += f" - Error: {run_data['last_error']}"
                        raise Exception(error_msg)
                    
                    attempts += 1
            except aiohttp.ClientError as e:
                print(f"Error checking run status: {e}")
                raise
        
        if attempts >= self.max_polling_attempts:
            raise Exception("Maximum polling attempts reached")

        # Get the assistant's response
        try:
            async with self.session.get(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
                messages = data["data"]
                
                # Find the assistant's message
                for message in messages:
                    if message["role"] == "assistant":
                        return message["content"][0]["text"]["value"]
                
                raise Exception("No assistant response found")
        except aiohttp.ClientError as e:
            print(f"Error fetching messages: {e}")
            raise
    
    async def generate_summary_and_score(self, company_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate one-line summary and score for an already-analyzed company"""
        # Format the prompt with analysis data
        prompt = SUMMARY_AND_SCORE_PROMPT.format(
            company_name=company_analysis.get("company_name", "Unknown"),
            evidence_of_early_pmf=company_analysis.get("evidence_of_early_pmf", "N/A"),
            incumbent_or_startup_alternatives_exist=company_analysis.get("incumbent_or_startup_alternatives_exist", "N/A"),
            founders_have_ft_work_experience=company_analysis.get("founders_have_ft_work_experience", "N/A"),
            founder_top_10_percent_distinction=company_analysis.get("founder_top_10_percent_distinction", "N/A"),
            founders_right_fit_for_product=company_analysis.get("founders_right_fit_for_product", "N/A"),
            problem_unique_and_specific=company_analysis.get("problem_unique_and_specific", "N/A"),
            opportunity_to_expand=company_analysis.get("opportunity_to_expand", "N/A"),
            tam_size=company_analysis.get("tam_size", "N/A"),
            direct_competitors=company_analysis.get("direct_competitors", "N/A"),
            founders_based_in_sf=company_analysis.get("founders_based_in_sf", "N/A")
        )
        
        try:
            # Create thread and run analysis
            thread_id = await self.create_thread()
            await self.add_message(thread_id, prompt)
            response = await self.run_assistant(thread_id)
            
            # Parse JSON response
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            result = json.loads(cleaned_response)
            
            # Validate the response
            if "one_line_summary" not in result or "company_score" not in result:
                raise ValueError("Missing required fields in response")
            
            # Ensure score is an integer between 1-5
            score = int(result["company_score"])
            if score < 1 or score > 5:
                raise ValueError(f"Score {score} is out of range (1-5)")
            
            return {
                "one_line_summary": result["one_line_summary"],
                "company_score": score
            }
            
        except Exception as e:
            print(f"Error generating summary/score for {company_analysis.get('company_name', 'Unknown')}: {e}")
            return {
                "one_line_summary": "Error generating summary",
                "company_score": 3  # Default to average
            }

    async def analyze_company(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a YC company using the OpenAI Assistant"""
        company_name = company_data.get("company_name", "Unknown")
        
        # Format the prompt with company data
        prompt = ANALYSIS_PROMPT.format(
            company_data=json.dumps(company_data, indent=2)
        )
        
        try:
            # Create thread and run analysis
            thread_id = await self.create_thread()
            await self.add_message(thread_id, prompt)
            response = await self.run_assistant(thread_id)
            
            # Parse the JSON response
            try:
                # Clean up the response - remove markdown code blocks if present
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                elif cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  # Remove ```
                
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove trailing ```
                
                cleaned_response = cleaned_response.strip()
                
                # Try to parse the cleaned response
                analysis_result = json.loads(cleaned_response)
                
            except json.JSONDecodeError as e:
                # If that fails, try to extract JSON from the response
                print(f"Error parsing response for {company_name}: {e}")
                
                # Look for JSON object in the response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    try:
                        analysis_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Return error structure
                        analysis_result = {
                            "evidence_of_early_pmf": f"Error parsing response for {company_name}",
                            "incumbent_or_startup_alternatives_exist": "Error parsing response",
                            "founders_have_ft_work_experience": "Error parsing response", 
                            "founder_top_10_percent_distinction": "Error parsing response",
                            "founders_right_fit_for_product": "Error parsing response",
                            "problem_unique_and_specific": "Error parsing response",
                            "opportunity_to_expand": "Error parsing response",
                            "tam_size": "Error parsing response",
                            "direct_competitors": "Error parsing response",
                            "founders_based_in_sf": "Error parsing response"
                        }
                else:
                    # Return error structure
                    analysis_result = {
                        "evidence_of_early_pmf": f"No JSON found in response for {company_name}",
                        "incumbent_or_startup_alternatives_exist": "Error parsing response",
                        "founders_have_ft_work_experience": "Error parsing response",
                        "founder_top_10_percent_distinction": "Error parsing response", 
                        "founders_right_fit_for_product": "Error parsing response",
                        "problem_unique_and_specific": "Error parsing response",
                        "opportunity_to_expand": "Error parsing response",
                        "tam_size": "Error parsing response",
                        "direct_competitors": "Error parsing response",
                        "founders_based_in_sf": "Error parsing response"
                    }
            
            # Add company name and timestamp
            analysis_result["company_name"] = company_name
            analysis_result["analysis_timestamp"] = datetime.now().isoformat()
            
            # Generate summary and score
            summary_score = await self.generate_summary_and_score(analysis_result)
            analysis_result["one_line_summary"] = summary_score["one_line_summary"]
            analysis_result["company_score"] = summary_score["company_score"]
            
            # Report progress
            if self.progress_callback:
                await self.progress_callback(company_name, "completed")
            
            return analysis_result
            
        except Exception as e:
            print(f"\nError analyzing {company_name}: {e}")
            
            # Report progress
            if self.progress_callback:
                await self.progress_callback(company_name, "failed")
            
            # Return error structure
            return {
                "company_name": company_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "evidence_of_early_pmf": f"Analysis failed: {str(e)}",
                "incumbent_or_startup_alternatives_exist": "Analysis failed",
                "founders_have_ft_work_experience": "Analysis failed",
                "founder_top_10_percent_distinction": "Analysis failed",
                "founders_right_fit_for_product": "Analysis failed",
                "problem_unique_and_specific": "Analysis failed",
                "opportunity_to_expand": "Analysis failed",
                "tam_size": "Analysis failed",
                "direct_competitors": "Analysis failed",
                "founders_based_in_sf": "Analysis failed"
            }


async def process_companies(companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple companies in parallel"""
    total_companies = len(companies)
    completed = 0
    failed = 0
    
    async def progress_callback(company_name: str, status: str):
        nonlocal completed, failed
        if status == "completed":
            completed += 1
        else:
            failed += 1
        
        print(f"\rProcessing {completed + failed}/{total_companies} companies... "
              f"(✓ {completed} completed, ✗ {failed} failed)", end="", flush=True)
    
    print(f"Starting parallel analysis of {total_companies} companies...")
    
    async with aiohttp.ClientSession() as session:
        analyzer = YCCompanyAnalyzer(session, progress_callback)
        
        # Create tasks for all companies
        tasks = [analyzer.analyze_company(company) for company in companies]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to proper results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # This shouldn't happen as we handle exceptions in analyze_company
                print(f"\nUnexpected error for {companies[i].get('company_name', 'Unknown')}: {result}")
            else:
                final_results.append(result)
    
    print(f"\n\nAnalysis complete! Processed {len(final_results)} companies.")
    return final_results


def save_results_to_csv(results: List[Dict[str, Any]], filename: str = "yc_company_analysis.csv") -> None:
    """Save all analysis results to CSV, replacing existing file"""
    if not results:
        print("No results to save.")
        return
    
    # Define the fieldnames in the order we want them
    fieldnames = [
        "company_name",
        "analysis_timestamp",
        "one_line_summary",
        "company_score",
        "evidence_of_early_pmf",
        "incumbent_or_startup_alternatives_exist",
        "founders_have_ft_work_experience",
        "founder_top_10_percent_distinction",
        "founders_right_fit_for_product",
        "problem_unique_and_specific",
        "opportunity_to_expand",
        "tam_size",
        "direct_competitors",
        "founders_based_in_sf"
    ]
    
    # Write to CSV (overwrite existing file)
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write all results
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to: {filename}")


async def add_summaries_and_scores(csv_file: str = "yc_company_analysis.csv") -> None:
    """Add one-line summaries and scores to existing analysis CSV"""
    # Read existing CSV
    companies = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        companies = list(reader)
    
    if not companies:
        print("No companies found in CSV file")
        return
    
    print(f"\nGenerating summaries and scores for {len(companies)} companies...")
    
    # Check if summaries/scores already exist
    if "one_line_summary" in companies[0] and "company_score" in companies[0]:
        print("Summaries and scores already exist in CSV. Overwriting...")
    
    async with aiohttp.ClientSession() as session:
        analyzer = YCCompanyAnalyzer(session)
        
        # Process each company
        for i, company in enumerate(companies):
            company_name = company.get("company_name", "Unknown")
            print(f"\rProcessing {i+1}/{len(companies)}: {company_name}...", end="", flush=True)
            
            # Generate summary and score
            result = await analyzer.generate_summary_and_score(company)
            
            # Add to company data
            company["one_line_summary"] = result["one_line_summary"]
            company["company_score"] = str(result["company_score"])
    
    print("\n\nWriting updated CSV...")
    
    # Define fieldnames (include new ones)
    fieldnames = [
        "company_name",
        "analysis_timestamp",
        "one_line_summary",
        "company_score",
        "evidence_of_early_pmf",
        "incumbent_or_startup_alternatives_exist",
        "founders_have_ft_work_experience",
        "founder_top_10_percent_distinction",
        "founders_right_fit_for_product",
        "problem_unique_and_specific",
        "opportunity_to_expand",
        "tam_size",
        "direct_competitors",
        "founders_based_in_sf"
    ]
    
    # Write updated CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(companies)
    
    print(f"Updated CSV saved to: {csv_file}")
    
    # Print score distribution
    score_counts = {}
    for company in companies:
        score = company.get("company_score", "3")
        score_counts[score] = score_counts.get(score, 0) + 1
    
    print("\nScore Distribution:")
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        percentage = (count / len(companies)) * 100
        print(f"  Score {score}: {count} companies ({percentage:.1f}%)")


async def main():
    """Main function to read company data and run parallel analysis"""
    import sys
    
    # Check if we're adding summaries to existing analysis
    if len(sys.argv) > 1 and sys.argv[1] == "--add-summaries":
        await add_summaries_and_scores()
        return
    
    print("YC Company Analysis Script (Parallel Version) Starting...")
    input_file = "company_input.json"
    
    # Read JSON from input file
    try:
        print(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            companies_data = json.load(f)
        
        # Ensure it's a list
        if not isinstance(companies_data, list):
            print("Error: Input file must contain an array of company objects.")
            sys.exit(1)
        
        print(f"Successfully loaded {len(companies_data)} companies")
        
        # Print company names
        for i, company in enumerate(companies_data, 1):
            print(f"  {i}. {company.get('company_name', 'Unknown')}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please create a file named 'company_input.json' with your companies data.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}'")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Process all companies in parallel
    print()  # Empty line for better formatting
    results = await process_companies(companies_data)
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    save_results_to_csv(results)
    
    print("\nAll done! ✨")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())