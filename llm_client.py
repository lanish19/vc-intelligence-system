#!/usr/bin/env python3
"""
LLM Client for Knowledge Extraction

This module implements LLM-based knowledge extraction using Google's Gemini API.
Based on the user's template and integrated with the VC intelligence system.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from config import get_config
from vc_prompts import VCPromptEngineering

logger = logging.getLogger(__name__)

class GeminiLLMClient:
    """
    LLM client for extracting VC knowledge using Google's Gemini API.
    
    Uses the exact model and configuration specified by the user:
    - Model: gemini-2.5-flash-lite-preview-06-17
    - API Key: GEMINI_API_KEY from environment
    """
    
    def __init__(self):
        self.config = get_config()
        self.client = genai.Client(
            api_key=self.config.llm.api_key
        )
        self.prompt_engine = VCPromptEngineering()
        
        logger.info(f"Initialized Gemini client with model: {self.config.llm.model}")
    
    def extract_knowledge(self, text: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Extract VC knowledge triplets from text using Gemini LLM.
        
        Args:
            text: Input text to analyze
            max_retries: Maximum retry attempts
            
        Returns:
            List of knowledge triplets in SPO format
        """
        
        # Get the extraction prompt
        extraction_prompt = self.prompt_engine.get_base_extraction_prompt()
        
        # Format the prompt with the input text
        formatted_prompt = extraction_prompt.user_prompt.format(text=text)
        
        # Prepare the full prompt
        full_prompt = f"{extraction_prompt.system_prompt}\n\n{formatted_prompt}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting knowledge extraction (attempt {attempt + 1}/{max_retries})")
                
                # Prepare content for Gemini
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=full_prompt),
                        ],
                    ),
                ]
                
                # Configure generation
                generate_content_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.config.llm.thinking_budget,
                    ),
                    response_mime_type="text/plain",
                    temperature=self.config.llm.temperature,
                    max_output_tokens=self.config.llm.max_tokens
                )
                
                # Generate content
                response_text = ""
                for chunk in self.client.models.generate_content_stream(
                    model=self.config.llm.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text:
                        response_text += chunk.text
                
                # Parse the JSON response
                triplets = self._parse_response(response_text)
                
                logger.info(f"Successfully extracted {len(triplets)} knowledge triplets")
                return triplets
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed")
                    return []
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        return []
    
    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response and extract JSON triplets"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                triplets = json.loads(json_str)
                
                # Validate and clean triplets
                return self._validate_triplets(triplets)
            else:
                logger.error("No valid JSON array found in response")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return []
    
    def _validate_triplets(self, triplets: List[Dict]) -> List[Dict[str, Any]]:
        """Validate and clean extracted triplets"""
        valid_triplets = []
        
        required_fields = ['subject', 'predicate', 'object', 'subject_type', 'object_type']
        
        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue
                
            # Check required fields
            if all(field in triplet for field in required_fields):
                # Add confidence if missing
                if 'confidence' not in triplet:
                    triplet['confidence'] = 0.8
                
                # Clean strings
                triplet['subject'] = str(triplet['subject']).strip()
                triplet['predicate'] = str(triplet['predicate']).strip()
                triplet['object'] = str(triplet['object']).strip()
                
                valid_triplets.append(triplet)
            else:
                logger.warning(f"Skipping invalid triplet: {triplet}")
        
        return valid_triplets
    
    def batch_extract(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Extract knowledge from multiple texts in batches.
        
        Args:
            texts: List of texts to process
            batch_size: Size of each batch (uses config default if None)
            
        Returns:
            List of triplet lists, one for each input text
        """
        if batch_size is None:
            batch_size = self.config.extraction.batch_size
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} texts)")
            
            batch_results = []
            for text in batch:
                triplets = self.extract_knowledge(text)
                batch_results.append(triplets)
            
            results.extend(batch_results)
            
            # Small delay between batches to be respectful to API
            if i + batch_size < len(texts):
                time.sleep(1)
        
        return results

class KnowledgeExtractor:
    """
    High-level interface for VC knowledge extraction.
    
    Integrates LLM client with the VC intelligence pipeline.
    """
    
    def __init__(self):
        self.llm_client = GeminiLLMClient()
    
    def extract_from_company_description(self, firm_name: str, description: str) -> List[Dict[str, Any]]:
        """
        Extract VC knowledge from a company description.
        
        Args:
            firm_name: Name of the company
            description: Company description text
            
        Returns:
            List of knowledge triplets
        """
        # Combine firm name and description for context
        full_text = f"Company: {firm_name}\n\nDescription: {description}"
        
        triplets = self.llm_client.extract_knowledge(full_text)
        
        # Ensure the company name is used consistently
        for triplet in triplets:
            if triplet['subject'].lower() in description.lower() or 'company' in triplet['subject'].lower():
                triplet['subject'] = firm_name
        
        return triplets
    
    def extract_from_csv_row(self, row_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract knowledge from a CSV row with structured company data.
        
        Args:
            row_data: Dictionary containing company data from CSV
            
        Returns:
            List of knowledge triplets
        """
        firm_name = row_data.get('firm_name', 'Unknown')
        description = row_data.get('Firm Description', '')
        
        # Build comprehensive text from all available fields
        text_parts = [f"Company: {firm_name}"]
        
        if description:
            text_parts.append(f"Description: {description}")
        
        if 'Primary Sector' in row_data:
            text_parts.append(f"Sector: {row_data['Primary Sector']}")
        
        if 'Problem Hypothesis' in row_data:
            text_parts.append(f"Problem: {row_data['Problem Hypothesis']}")
        
        if 'Firm Unique Differentiation' in row_data:
            text_parts.append(f"Differentiation: {row_data['Firm Unique Differentiation']}")
        
        ai_role_key = 'Role of AI In Firm\'s Business Model and/or Offering'
        if ai_role_key in row_data:
            text_parts.append(f"AI Role: {row_data[ai_role_key]}")
        
        full_text = "\n\n".join(text_parts)
        
        return self.llm_client.extract_knowledge(full_text)

if __name__ == "__main__":
    # Test the LLM client
    try:
        extractor = KnowledgeExtractor()
        
        test_text = """
        CyberGuard provides AI-powered threat detection for enterprise networks. 
        Their proprietary behavioral analysis reduces false positives by 90% compared to traditional SIEM solutions. 
        CyberGuard targets mid-market financial institutions and competes directly with CrowdStrike and SentinelOne.
        """
        
        triplets = extractor.llm_client.extract_knowledge(test_text)
        
        print(f"Extracted {len(triplets)} triplets:")
        for triplet in triplets:
            print(f"  {triplet}")
            
    except Exception as e:
        print(f"Error testing LLM client: {e}")
        print("Make sure GEMINI_API_KEY is set in your environment!") 