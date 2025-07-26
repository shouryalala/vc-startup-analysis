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


async def main():
    """Main function to read company data and run parallel analysis"""
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