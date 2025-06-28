#!/usr/bin/env python3
"""
CSV Knowledge Extractor for Venture Capital Intelligence

This module implements LLM-based knowledge extraction from CSV data containing startup/company 
information. It uses Google's Gemini API to extract Subject-Predicate-Object triplets following 
the bespoke VC ontology.

Author: AI Mapping Knowledge Graph System
"""

import pandas as pd
import json
import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import unicodedata
import string
from datetime import datetime
from llm_client import KnowledgeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SPOTriplet:
    """Subject-Predicate-Object triplet following the VC ontology"""
    subject: str
    predicate: str
    object: str
    subject_type: str
    object_type: str
    confidence: float
    source_field: str
    source_row: int
    extracted_at: str

class VCOntologyExtractor:
    """
    LLM-powered extractor for VC knowledge from CSV data.
    
    Uses Google's Gemini API to extract knowledge according to the VC-specific ontology:
    - Company: The startup/firm being analyzed
    - Technology: Technical approaches, algorithms, platforms
    - Market: Target markets, customer segments
    - MarketFriction: Problems, pain points, market gaps
    - Differentiator: Unique value propositions, competitive advantages
    - ThesisConcept: High-level investment theses and assumptions
    """
    
    def __init__(self):
        self.triplets = []
        self.knowledge_extractor = KnowledgeExtractor()
        
        logger.info("Initialized VCOntologyExtractor with LLM-based extraction")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize unicode
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char in string.printable)
        
        return text

    def extract_from_row(self, row: pd.Series) -> List[SPOTriplet]:
        """
        Extract all knowledge from a CSV row using LLM.
        
        This replaces all the individual rule-based extraction methods.
        """
        triplets = []
        firm_name = self.clean_text(str(row.get('firm_name', '')))
        
        if not firm_name:
            logger.warning(f"Skipping row {row.name} - no firm name")
            return triplets
        
        # Convert row to dictionary for LLM processing
        row_data = row.to_dict()
        
        try:
            # Use LLM to extract knowledge triplets
            llm_triplets = self.knowledge_extractor.extract_from_csv_row(row_data)
            
            # Convert LLM triplets to our SPOTriplet format
            for i, triplet in enumerate(llm_triplets):
                try:
                    spo_triplet = SPOTriplet(
                        subject=triplet['subject'],
                        predicate=triplet['predicate'],
                        object=triplet['object'],
                        subject_type=triplet['subject_type'],
                        object_type=triplet['object_type'],
                        confidence=triplet.get('confidence', 0.8),
                        source_field="LLM_extraction",
                        source_row=row.name,
                        extracted_at=datetime.now().isoformat()
                    )
                    triplets.append(spo_triplet)
                    
                except KeyError as e:
                    logger.warning(f"Skipping malformed triplet {i} from row {row.name}: missing {e}")
                    continue
            
            logger.info(f"Extracted {len(triplets)} triplets from {firm_name}")
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge from {firm_name}: {str(e)}")
        
        return triplets

    def extract_structured_fields(self, row: pd.Series) -> List[SPOTriplet]:
        """
        Extract knowledge from structured fields (autonomy score, etc.) using rule-based approach.
        These are supplementary to the LLM extraction.
        """
        triplets = []
        firm_name = self.clean_text(str(row.get('firm_name', '')))
        
        if not firm_name:
            return triplets
        
        # Autonomy spectrum - using a simplified column reference
        autonomy_col = None
        for col in row.index:
            if 'Autonomy Spectrum' in str(col):
                autonomy_col = col
                break
        
        if autonomy_col and pd.notna(row.get(autonomy_col)):
            autonomy_score = row.get(autonomy_col)
            try:
                score = float(autonomy_score)
                triplets.append(SPOTriplet(
                    subject=firm_name,
                    predicate="has_autonomy_level",
                    object=f"autonomy_level_{score:.1f}",
                    subject_type="Company",
                    object_type="Technology",
                    confidence=0.95,
                    source_field="Autonomy Spectrum",
                    source_row=row.name,
                    extracted_at=datetime.now().isoformat()
                ))
            except (ValueError, TypeError):
                logger.warning(f"Invalid autonomy score for {firm_name}: {autonomy_score}")
        
        # Deployment spectrum
        deployment_col = None
        for col in row.index:
            if 'Deployment Spectrum' in str(col):
                deployment_col = col
                break
        
        if deployment_col and pd.notna(row.get(deployment_col)):
            deployment_score = row.get(deployment_col)
            try:
                score = float(deployment_score)
                deployment_type = ("tactical_edge" if score >= 7 
                                 else "cloud_enterprise" if score <= 3 
                                 else "hybrid_deployment")
                
                triplets.append(SPOTriplet(
                    subject=firm_name,
                    predicate="deploys_on",
                    object=deployment_type,
                    subject_type="Company",
                    object_type="Technology",
                    confidence=0.95,
                    source_field="Deployment Spectrum",
                    source_row=row.name,
                    extracted_at=datetime.now().isoformat()
                ))
            except (ValueError, TypeError):
                logger.warning(f"Invalid deployment score for {firm_name}: {deployment_score}")
        
        return triplets

    def process_csv(self, csv_path: str, use_llm: bool = True, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process entire CSV file and extract knowledge triplets.
        
        Args:
            csv_path: Path to the CSV file
            use_llm: Whether to use LLM-based extraction (default True)
            max_rows: Maximum number of rows to process (for testing)
        """
        logger.info(f"Processing CSV file: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            if max_rows:
                df = df.head(max_rows)
                logger.info(f"Limited to {max_rows} rows for processing")
            
            all_triplets = []
            
            for idx, row in df.iterrows():
                logger.info(f"Processing row {idx + 1}/{len(df)}")
                
                # Extract using LLM
                if use_llm:
                    llm_triplets = self.extract_from_row(row)
                    all_triplets.extend(llm_triplets)
                
                # Extract structured fields (always)
                structured_triplets = self.extract_structured_fields(row)
                all_triplets.extend(structured_triplets)
            
            logger.info(f"Total triplets extracted: {len(all_triplets)}")
            
            # Convert to dictionaries for JSON serialization
            return [asdict(triplet) for triplet in all_triplets]
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    def save_triplets(self, triplets: List[Dict[str, Any]], output_path: str):
        """Save triplets to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(triplets)} triplets to {output_path}")

    def get_extraction_stats(self, triplets: List[SPOTriplet]) -> Dict[str, Any]:
        """Generate statistics about the extracted triplets"""
        if not triplets:
            return {"total": 0}
        
        predicates = [t.predicate for t in triplets]
        subject_types = [t.subject_type for t in triplets]
        object_types = [t.object_type for t in triplets]
        
        return {
            "total_triplets": len(triplets),
            "unique_subjects": len(set(t.subject for t in triplets)),
            "unique_predicates": len(set(predicates)),
            "unique_objects": len(set(t.object for t in triplets)),
            "predicate_distribution": {pred: predicates.count(pred) for pred in set(predicates)},
            "subject_type_distribution": {st: subject_types.count(st) for st in set(subject_types)},
            "object_type_distribution": {ot: object_types.count(ot) for ot in set(object_types)},
            "average_confidence": sum(t.confidence for t in triplets) / len(triplets)
        }

def main():
    """Example usage of the LLM-powered VC ontology extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract VC knowledge from CSV using LLM')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('output_json', help='Path to output JSON file')
    parser.add_argument('--max-rows', type=int, help='Maximum rows to process (for testing)')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM extraction (structured only)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = VCOntologyExtractor()
    
    # Process CSV
    try:
        triplets = extractor.process_csv(
            args.input_csv,
            use_llm=not args.no_llm,
            max_rows=args.max_rows
        )
        
        # Save results
        extractor.save_triplets(triplets, args.output_json)
        
        # Print statistics
        stats = extractor.get_extraction_stats([SPOTriplet(**t) for t in triplets])
        print("\nExtraction Statistics:")
        print(f"Total triplets: {stats['total_triplets']}")
        print(f"Unique subjects: {stats['unique_subjects']}")
        print(f"Average confidence: {stats['average_confidence']:.2f}")
        print(f"Top predicates: {dict(list(stats['predicate_distribution'].items())[:5])}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 