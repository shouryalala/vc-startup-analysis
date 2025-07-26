#!/usr/bin/env python3
"""
YC Company Analysis Script
Analyzes YC company data using OpenAI Assistant API
"""

import json
import sys
import time
import csv
import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional
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
    def __init__(self):
        self.api_key = API_KEY
        self.assistant_id = ASSISTANT_ID
        self.base_url = API_BASE_URL
        self.headers = HEADERS
        self.max_polling_attempts = 60
        self.polling_interval = 2

    def create_thread(self) -> str:
        """Create a new conversation thread"""
        print("Creating thread...")
        url = f"{self.base_url}/threads"
        print(f"POST {url}")
        
        try:
            response = requests.post(url, headers=self.headers, json={})
            response.raise_for_status()
            thread_id = response.json()["id"]
            print(f"Thread created successfully: {thread_id}")
            return thread_id
        except requests.exceptions.RequestException as e:
            print(f"Error creating thread: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise

    def add_message(self, thread_id: str, content: str) -> None:
        """Add a message to the thread"""
        print(f"Adding message to thread {thread_id}...")
        url = f"{self.base_url}/threads/{thread_id}/messages"
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json={
                    "role": "user",
                    "content": content
                }
            )
            response.raise_for_status()
            print("Message added successfully")
        except requests.exceptions.RequestException as e:
            print(f"Error adding message: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise

    def run_assistant(self, thread_id: str) -> str:
        """Run the assistant and get the response"""
        print(f"Starting assistant run for thread {thread_id}...")
        url = f"{self.base_url}/threads/{thread_id}/runs"
        
        # Start the run
        try:
            print(f"Using assistant ID: {self.assistant_id}")
            print(f"Using model: {MODEL}")
            
            # Try without model parameter first
            response = requests.post(
                url,
                headers=self.headers,
                json={
                    "assistant_id": self.assistant_id
                }
            )
            
            if response.status_code != 200:
                print(f"Error response: {response.status_code}")
                print(f"Error details: {response.text}")
                response.raise_for_status()
                
            run_id = response.json()["id"]
            print(f"Run started successfully: {run_id}")
        except requests.exceptions.RequestException as e:
            print(f"Error starting run: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise

        # Poll for completion
        attempts = 0
        print(f"Polling for completion (max {self.max_polling_attempts} attempts)...")
        
        while attempts < self.max_polling_attempts:
            time.sleep(self.polling_interval)
            
            # Check run status
            try:
                response = requests.get(
                    f"{self.base_url}/threads/{thread_id}/runs/{run_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                run_data = response.json()
                
                status = run_data["status"]
                print(f"Attempt {attempts + 1}: Run status = {status}")
                
                if status == "completed":
                    print("Run completed successfully!")
                    break
                elif status in ["failed", "cancelled", "expired"]:
                    error_msg = f"Run failed with status: {status}"
                    if "last_error" in run_data and run_data["last_error"]:
                        error_msg += f" - Error: {run_data['last_error']}"
                    raise Exception(error_msg)
                
                attempts += 1
            except requests.exceptions.RequestException as e:
                print(f"Error checking run status: {e}")
                raise
        
        if attempts >= self.max_polling_attempts:
            raise Exception("Maximum polling attempts reached")

        # Get the assistant's response
        print("Fetching assistant's response...")
        try:
            response = requests.get(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=self.headers
            )
            response.raise_for_status()
            messages = response.json()["data"]
            
            print(f"Found {len(messages)} messages in thread")
            
            # Find the assistant's message
            for message in messages:
                if message["role"] == "assistant":
                    content = message["content"][0]["text"]["value"]
                    print(f"Assistant response length: {len(content)} characters")
                    return content
            
            raise Exception("No assistant response found")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching messages: {e}")
            raise

    def analyze_company(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a YC company using the OpenAI Assistant"""
        print("\n=== Starting Company Analysis ===")
        
        # Format the prompt with company data
        prompt = ANALYSIS_PROMPT.format(
            company_data=json.dumps(company_data, indent=2)
        )
        print(f"Prompt length: {len(prompt)} characters")
        
        try:
            # Create thread and run analysis
            thread_id = self.create_thread()
            self.add_message(thread_id, prompt)
            response = self.run_assistant(thread_id)
        except Exception as e:
            print(f"\nError during API interaction: {e}")
            raise
        
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
            print(f"Initial parsing failed: {e}")
            print("Attempting to extract JSON from response...")
            
            # Look for JSON object in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    analysis_result = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"JSON extraction also failed: {e2}")
                    print(f"Raw response:\n{response}")
                    # Return a default structure if parsing fails
                    analysis_result = {
                        "evidence_of_early_pmf": "Error parsing response - check console output",
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
                print("No JSON object found in response")
                print(f"Raw response:\n{response}")
                # Return a default structure
                analysis_result = {
                    "evidence_of_early_pmf": "No JSON found in assistant response",
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
        analysis_result["company_name"] = company_data.get("company_name", "Unknown")
        analysis_result["analysis_timestamp"] = datetime.now().isoformat()
        
        return analysis_result

    def save_to_csv(self, analysis_result: Dict[str, Any], filename: str = "yc_company_analysis.csv") -> None:
        """Save analysis results to CSV"""
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
        
        # Check if file exists
        file_exists = False
        try:
            with open(filename, 'r') as f:
                file_exists = True
        except FileNotFoundError:
            pass
        
        # Write to CSV
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write the data
            writer.writerow(analysis_result)


def main():
    """Main function to read company data from input file"""
    print("YC Company Analysis Script Starting...")
    input_file = "company_input.json"
    
    # Read JSON from input file
    try:
        print(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            company_data = json.load(f)
        print("Successfully loaded company data")
        print(f"Company name: {company_data.get('company_name', 'Unknown')}")
        print(f"Number of founders: {len(company_data.get('founders', []))}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please create a file named 'company_input.json' with your company data.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}'")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Create analyzer and run analysis
    print("\nInitializing YC Company Analyzer...")
    analyzer = YCCompanyAnalyzer()
    
    print(f"\nAnalyzing {company_data.get('company_name', 'Unknown Company')}...")
    
    try:
        # Run analysis
        result = analyzer.analyze_company(company_data)
        
        # Save to CSV
        print("\nSaving results to CSV...")
        analyzer.save_to_csv(result)
        
        # Print results
        print("\nAnalysis Complete!")
        print(f"Results saved to: yc_company_analysis.csv")
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()