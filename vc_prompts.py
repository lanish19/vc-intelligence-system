#!/usr/bin/env python3
"""
VC Prompt Engineering System

This module implements Section 2 of the strategic framework: "Precision Prompt Engineering 
for Venture Capital Intelligence". It provides a bespoke ontology and sophisticated prompt 
templates for extracting VC-relevant knowledge from pitch documents.

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Container for a prompt template with metadata"""
    name: str
    version: str
    system_prompt: str
    user_prompt: str
    examples: List[Dict[str, Any]]

@dataclass
class ExtractionPrompt:
    """Structured prompt for knowledge extraction"""
    system_prompt: str
    user_prompt: str
    few_shot_examples: List[Dict[str, Any]]
    extraction_rules: List[str]
    ontology_types: Dict[str, str]

class VCPrompts:
    """
    VC-specific prompt templates following the strategic framework.
    
    Implements:
    - Bespoke VC ontology
    - Few-shot prompting
    - Precise extraction rules
    """
    
    def __init__(self):
        self.ontology = self._define_ontology()
    
    def _define_ontology(self) -> Dict[str, Any]:
        """Define the VC ontology from Section 2.3"""
        return {
            "entities": ["Company", "Technology", "Market", "MarketFriction", "Differentiator", "ThesisConcept"],
            "relationships": ["has_thesis", "addresses_friction", "has_differentiator", "targets_market", "uses_technology", "competes_with"]
        }
    
    def get_extraction_prompt(self) -> str:
        """Get the main extraction prompt"""
        return """You are a venture capital analyst extracting knowledge from pitch documents.

Extract Subject-Predicate-Object triplets following this VC ontology:

ENTITY TYPES:
- Company: The startup being analyzed
- Technology: Technical approaches, algorithms, platforms  
- Market: Target markets, customer segments
- MarketFriction: Problems, pain points, market gaps
- Differentiator: Unique advantages, competitive moats
- ThesisConcept: Investment theses, strategic assumptions

EXTRACTION RULES:
1. Predicates must be 1-3 words maximum
2. Replace pronouns with actual entity names
3. Use consistent entity naming
4. Output valid JSON array format

VC-SPECIFIC RULES:
- Investment Thesis: {"subject": "company", "predicate": "has_thesis", "object": "thesis_concept", "subject_type": "Company", "object_type": "ThesisConcept"}
- Market Friction: {"subject": "company", "predicate": "addresses_friction", "object": "problem", "subject_type": "Company", "object_type": "MarketFriction"}  
- Differentiators: {"subject": "company", "predicate": "has_differentiator", "object": "advantage", "subject_type": "Company", "object_type": "Differentiator"}
- Competition: {"subject": "company", "predicate": "competes_with", "object": "competitor", "subject_type": "Company", "object_type": "Company"}
- Target Market: {"subject": "company", "predicate": "targets_market", "object": "market", "subject_type": "Company", "object_type": "Market"}
- Technology: {"subject": "company", "predicate": "uses_technology", "object": "tech", "subject_type": "Company", "object_type": "Technology"}

EXAMPLE:
Input: "Acme Security uses machine learning for autonomous threat detection. We compete with CrowdStrike in enterprise security."
Output: [
  {"subject": "Acme Security", "predicate": "uses_technology", "object": "machine learning", "subject_type": "Company", "object_type": "Technology"},
  {"subject": "Acme Security", "predicate": "competes_with", "object": "CrowdStrike", "subject_type": "Company", "object_type": "Company"},
  {"subject": "Acme Security", "predicate": "targets_market", "object": "enterprise security", "subject_type": "Company", "object_type": "Market"}
]

Extract knowledge triplets from this text:
"""

class VCPromptEngineering:
    """
    Provides precision prompts for VC-specific knowledge extraction.
    
    Implements the bespoke ontology:
    - Company, Technology, Market, MarketFriction, Differentiator, ThesisConcept
    """
    
    def __init__(self):
        self.ontology_types = {
            'Company': 'The startup or firm being analyzed',
            'Technology': 'Technical solutions, platforms, algorithms, or capabilities',
            'Market': 'Target customer segments, use cases, or market verticals',
            'MarketFriction': 'Problems, pain points, or customer needs being addressed',
            'Differentiator': 'Unique advantages, competitive moats, or value propositions',
            'ThesisConcept': 'Investment hypotheses, strategic beliefs, or market assumptions',
            'Competitor': 'Named competitive companies or alternative solutions',
            'Sector': 'Industry vertical or domain focus'
        }
        
        self.predicate_types = {
            'has_thesis': 'Connects company to its core investment hypothesis',
            'addresses_friction': 'Links company to problems it solves',
            'has_differentiator': 'Associates company with unique advantages',
            'targets_market': 'Defines company\'s market focus',
            'uses_technology': 'Describes technologies employed',
            'competes_with': 'Identifies competitive relationships',
            'operates_in': 'Defines sector or industry',
            'has_capability': 'Describes functional capabilities',
            'enables': 'Shows enabling relationships',
            'reduces': 'Quantifies reduction benefits',
            'improves': 'Quantifies improvement benefits'
        }
    
    def get_base_extraction_prompt(self) -> ExtractionPrompt:
        """
        Base prompt for extracting VC knowledge from text.
        Implements the strategic framework's prompt engineering guidelines.
        """
        
        system_prompt = """You are a specialized VC Analyst AI tasked with extracting structured investment intelligence from startup descriptions. Your role is to identify and structure information according to a precise venture capital ontology.

You must extract knowledge as Subject-Predicate-Object (SPO) triplets that capture investment-relevant relationships. Focus on identifying:
- Investment theses and strategic hypotheses
- Market frictions and customer pain points  
- Unique differentiators and competitive advantages
- Target markets and customer segments
- Technologies and capabilities
- Competitive landscape

Your output must be precise, factual, and directly traceable to the input text."""
        
        extraction_rules = [
            "RULE 1: Extract only factual information directly stated or clearly implied in the text",
            "RULE 2: Predicates must be 1-3 words maximum (e.g., 'has_thesis', 'addresses_friction')",
            "RULE 3: Replace all pronouns with actual entity names they refer to",
            "RULE 4: Output must be valid JSON array of objects with required fields",
            "RULE 5: Use only allowed entity types: Company, Technology, Market, MarketFriction, Differentiator, ThesisConcept, Competitor, Sector",
            "RULE 6: Use only allowed predicates: has_thesis, addresses_friction, has_differentiator, targets_market, uses_technology, competes_with, operates_in, has_capability, enables, reduces, improves",
            "RULE 7: Investment Thesis: Model as Company -[has_thesis]-> ThesisConcept. Example: {'subject': 'Acme Corp', 'predicate': 'has_thesis', 'object': 'future of enterprise security is autonomous'}",
            "RULE 8: Market Friction: Model as Company -[addresses_friction]-> MarketFriction. Example: {'subject': 'Acme Corp', 'predicate': 'addresses_friction', 'object': 'high cost and complexity of SIEM solutions'}",
            "RULE 9: Differentiators: Model as Company -[has_differentiator]-> Differentiator. Example: {'subject': 'Acme Corp', 'predicate': 'has_differentiator', 'object': 'patented quantum-resistant encryption'}",
            "RULE 10: Competitors: Model as Company -[competes_with]-> Competitor. Example: {'subject': 'Acme Corp', 'predicate': 'competes_with', 'object': 'CrowdStrike'}",
            "RULE 11: Target Market: Model as Company -[targets_market]-> Market. Example: {'subject': 'Acme Corp', 'predicate': 'targets_market', 'object': 'mid-market financial institutions'}",
            "RULE 12: Each triplet must include: subject, predicate, object, subject_type, object_type, confidence (0.0-1.0)"
        ]
        
        user_prompt = """Extract venture capital intelligence from the following company description. Return structured knowledge as JSON triplets following the exact format specified.

Focus on:
1. Core investment thesis or strategic hypothesis
2. Market problems/frictions being addressed
3. Unique differentiators or competitive advantages
4. Target markets and customer segments
5. Key technologies and capabilities
6. Named competitors

Company Description:
{text}

Return only valid JSON array with this exact structure:
[
  {
    "subject": "entity_name",
    "predicate": "relationship_type", 
    "object": "related_entity",
    "subject_type": "entity_type",
    "object_type": "entity_type",
    "confidence": 0.85
  }
]"""
        
        few_shot_examples = [
            {
                "input": "CyberGuard provides AI-powered threat detection for enterprise networks. Their proprietary behavioral analysis reduces false positives by 90% compared to traditional SIEM solutions. CyberGuard targets mid-market financial institutions and competes directly with CrowdStrike and SentinelOne.",
                "output": [
                    {
                        "subject": "CyberGuard",
                        "predicate": "uses_technology",
                        "object": "AI-powered threat detection",
                        "subject_type": "Company",
                        "object_type": "Technology",
                        "confidence": 0.95
                    },
                    {
                        "subject": "CyberGuard", 
                        "predicate": "has_differentiator",
                        "object": "90% reduction in false positives",
                        "subject_type": "Company",
                        "object_type": "Differentiator",
                        "confidence": 0.90
                    },
                    {
                        "subject": "CyberGuard",
                        "predicate": "targets_market", 
                        "object": "mid-market financial institutions",
                        "subject_type": "Company",
                        "object_type": "Market",
                        "confidence": 0.85
                    },
                    {
                        "subject": "CyberGuard",
                        "predicate": "competes_with",
                        "object": "CrowdStrike",
                        "subject_type": "Company", 
                        "object_type": "Competitor",
                        "confidence": 0.95
                    }
                ]
            },
            {
                "input": "Defense contractor AeroTech believes autonomous drones will transform battlefield operations. They address the military's challenge of pilot shortage and dangerous reconnaissance missions through AI-powered unmanned systems.",
                "output": [
                    {
                        "subject": "AeroTech",
                        "predicate": "has_thesis",
                        "object": "autonomous drones will transform battlefield operations", 
                        "subject_type": "Company",
                        "object_type": "ThesisConcept",
                        "confidence": 0.90
                    },
                    {
                        "subject": "AeroTech",
                        "predicate": "addresses_friction",
                        "object": "military pilot shortage",
                        "subject_type": "Company",
                        "object_type": "MarketFriction", 
                        "confidence": 0.85
                    },
                    {
                        "subject": "AeroTech",
                        "predicate": "addresses_friction",
                        "object": "dangerous reconnaissance missions",
                        "subject_type": "Company",
                        "object_type": "MarketFriction",
                        "confidence": 0.85
                    },
                    {
                        "subject": "AeroTech",
                        "predicate": "uses_technology",
                        "object": "AI-powered unmanned systems",
                        "subject_type": "Company",
                        "object_type": "Technology",
                        "confidence": 0.90
                    }
                ]
            }
        ]
        
        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            few_shot_examples=few_shot_examples,
            extraction_rules=extraction_rules,
            ontology_types=self.ontology_types
        )
    
    def get_thesis_extraction_prompt(self) -> ExtractionPrompt:
        """Specialized prompt for extracting investment theses"""
        
        system_prompt = """You are a VC investment thesis analyst. Extract and structure investment hypotheses and strategic beliefs from startup descriptions. Focus on identifying the core assumptions about market evolution, technology trends, and customer behavior that underpin the investment case."""
        
        user_prompt = """Extract investment theses from this company description. Look for statements about:
- Market evolution beliefs ("We believe X market will grow...")
- Technology trend predictions ("The future of Y is Z...")  
- Customer behavior assumptions ("Enterprises are moving toward...")
- Strategic hypotheses about market direction

Company Description:
{text}

Return JSON array of thesis triplets:
[{"subject": "company", "predicate": "has_thesis", "object": "thesis_statement", "subject_type": "Company", "object_type": "ThesisConcept", "confidence": 0.0-1.0}]"""
        
        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            few_shot_examples=[],
            extraction_rules=["Focus only on investment theses and strategic hypotheses"],
            ontology_types=self.ontology_types
        )
    
    def get_competitive_analysis_prompt(self) -> ExtractionPrompt:
        """Specialized prompt for competitive landscape extraction"""
        
        system_prompt = """You are a competitive intelligence analyst. Extract information about competitive positioning, named competitors, and market comparisons from startup descriptions."""
        
        user_prompt = """Extract competitive intelligence from this company description. Look for:
- Named competitors and companies  
- Competitive positioning statements
- Market comparisons and alternatives
- Differentiation vs competitors

Company Description:
{text}

Return JSON array focusing on competitive relationships:
[{"subject": "company", "predicate": "competes_with", "object": "competitor_name", "subject_type": "Company", "object_type": "Competitor", "confidence": 0.0-1.0}]"""
        
        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            few_shot_examples=[],
            extraction_rules=["Extract only named competitors and explicit competitive relationships"],
            ontology_types=self.ontology_types
        )
    
    def get_market_friction_prompt(self) -> ExtractionPrompt:
        """Specialized prompt for market friction and problem extraction"""
        
        system_prompt = """You are a market problem analyst. Extract and structure customer pain points, market frictions, and problems being solved by startups. Focus on identifying the specific challenges customers face."""
        
        user_prompt = """Extract market frictions and customer problems from this company description. Look for:
- Customer pain points and challenges
- Market inefficiencies being addressed  
- Problems with current solutions
- Unmet needs and gaps

Company Description:
{text}

Return JSON array of friction/problem relationships:
[{"subject": "company", "predicate": "addresses_friction", "object": "problem_description", "subject_type": "Company", "object_type": "MarketFriction", "confidence": 0.0-1.0}]"""
        
        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            few_shot_examples=[],
            extraction_rules=["Focus only on problems, pain points, and market frictions"],
            ontology_types=self.ontology_types
        )
    
    def get_differentiator_prompt(self) -> ExtractionPrompt:
        """Specialized prompt for differentiator and advantage extraction"""
        
        system_prompt = """You are a competitive advantage analyst. Extract and structure unique differentiators, competitive moats, and value propositions from startup descriptions. Focus on what makes each company unique."""
        
        user_prompt = """Extract differentiators and competitive advantages from this company description. Look for:
- Unique selling propositions (USPs)
- Competitive moats and barriers
- Quantified benefits and improvements
- Proprietary technologies or approaches  
- Performance metrics vs alternatives

Company Description:
{text}

Return JSON array of differentiator relationships:
[{"subject": "company", "predicate": "has_differentiator", "object": "advantage_description", "subject_type": "Company", "object_type": "Differentiator", "confidence": 0.0-1.0}]"""
        
        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt, 
            few_shot_examples=[],
            extraction_rules=["Focus only on unique advantages and differentiators"],
            ontology_types=self.ontology_types
        )
    
    def get_technology_extraction_prompt(self) -> ExtractionPrompt:
        """Specialized prompt for technology and capability extraction"""
        
        system_prompt = """You are a technology analyst. Extract and structure technologies, platforms, and capabilities from startup descriptions. Focus on technical solutions and implementation approaches."""
        
        user_prompt = """Extract technologies and capabilities from this company description. Look for:
- Core technologies and platforms
- AI/ML techniques and algorithms
- Technical capabilities and features
- Implementation approaches
- Technical infrastructure

Company Description:
{text}

Return JSON array of technology relationships:
[{"subject": "company", "predicate": "uses_technology", "object": "technology_name", "subject_type": "Company", "object_type": "Technology", "confidence": 0.0-1.0}]"""
        
        return ExtractionPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            few_shot_examples=[],
            extraction_rules=["Focus only on technologies, platforms, and technical capabilities"],
            ontology_types=self.ontology_types
        )
    
    def format_prompt_for_llm(self, prompt: ExtractionPrompt, text: str, 
                             include_examples: bool = True) -> Dict[str, str]:
        """Format prompt for LLM API calls"""
        
        messages = [
            {"role": "system", "content": prompt.system_prompt}
        ]
        
        # Add few-shot examples if requested
        if include_examples and prompt.few_shot_examples:
            examples_text = "\n\nExamples of correct extraction:\n\n"
            for i, example in enumerate(prompt.few_shot_examples, 1):
                examples_text += f"Example {i}:\n"
                examples_text += f"Input: {example['input']}\n"
                examples_text += f"Output: {json.dumps(example['output'], indent=2)}\n\n"
            
            messages[0]["content"] += examples_text
        
        # Add extraction rules
        rules_text = "\n\nExtraction Rules:\n"
        for rule in prompt.extraction_rules:
            rules_text += f"- {rule}\n"
        messages[0]["content"] += rules_text
        
        # Add ontology definitions
        ontology_text = "\n\nEntity Type Definitions:\n"
        for entity_type, description in prompt.ontology_types.items():
            ontology_text += f"- {entity_type}: {description}\n"
        messages[0]["content"] += ontology_text
        
        # Add user message with text to analyze
        user_content = prompt.user_prompt.format(text=text)
        messages.append({"role": "user", "content": user_content})
        
        return {"messages": messages}
    
    def validate_extraction_output(self, output: List[Dict], 
                                 allowed_predicates: List[str] = None) -> Dict[str, Any]:
        """Validate extracted triplets against ontology rules"""
        
        if allowed_predicates is None:
            allowed_predicates = list(self.predicate_types.keys())
        
        validation_results = {
            "valid_triplets": [],
            "invalid_triplets": [],
            "errors": [],
            "stats": {
                "total": len(output),
                "valid": 0,
                "invalid": 0
            }
        }
        
        required_fields = {"subject", "predicate", "object", "subject_type", "object_type", "confidence"}
        
        for i, triplet in enumerate(output):
            errors = []
            
            # Check required fields
            missing_fields = required_fields - set(triplet.keys())
            if missing_fields:
                errors.append(f"Missing fields: {missing_fields}")
            
            # Check predicate validity
            if triplet.get("predicate") not in allowed_predicates:
                errors.append(f"Invalid predicate: {triplet.get('predicate')}")
            
            # Check entity types
            if triplet.get("subject_type") not in self.ontology_types:
                errors.append(f"Invalid subject_type: {triplet.get('subject_type')}")
            
            if triplet.get("object_type") not in self.ontology_types:
                errors.append(f"Invalid object_type: {triplet.get('object_type')}")
            
            # Check confidence range
            confidence = triplet.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errors.append(f"Invalid confidence: {confidence} (must be 0.0-1.0)")
            
            if errors:
                validation_results["invalid_triplets"].append({
                    "index": i,
                    "triplet": triplet,
                    "errors": errors
                })
                validation_results["stats"]["invalid"] += 1
            else:
                validation_results["valid_triplets"].append(triplet)
                validation_results["stats"]["valid"] += 1
        
        return validation_results

def main():
    """Demo the prompt engineering system"""
    prompt_engine = VCPromptEngineering()
    
    # Get base extraction prompt
    base_prompt = prompt_engine.get_base_extraction_prompt()
    
    # Example text
    example_text = """
    DefenseAI provides autonomous threat detection for military networks using proprietary 
    behavioral analysis algorithms. Their system reduces false positives by 95% compared to 
    traditional solutions and operates in real-time at the tactical edge. DefenseAI targets 
    DoD contractors and competes with Palantir and Raytheon. They believe the future of 
    cyber defense is fully autonomous AI that can respond to threats faster than human analysts.
    """
    
    # Format for LLM
    formatted_prompt = prompt_engine.format_prompt_for_llm(base_prompt, example_text)
    
    print("=== VC PROMPT ENGINEERING SYSTEM ===")
    print("\nSystem Prompt:")
    print(formatted_prompt["messages"][0]["content"][:500] + "...")
    print("\nUser Prompt:")
    print(formatted_prompt["messages"][1]["content"][:500] + "...")
    
    print(f"\nOntology Types: {len(prompt_engine.ontology_types)}")
    for entity_type, description in prompt_engine.ontology_types.items():
        print(f"  {entity_type}: {description}")
    
    print(f"\nPredicate Types: {len(prompt_engine.predicate_types)}")
    for predicate, description in prompt_engine.predicate_types.items():
        print(f"  {predicate}: {description}")

if __name__ == "__main__":
    main() 