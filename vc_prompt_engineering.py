#!/usr/bin/env python3
"""
VC Prompt Engineering System

This module implements Section 2 of the strategic framework: "Precision Prompt Engineering 
for Venture Capital Intelligence". It provides a bespoke ontology and sophisticated prompt 
templates for extracting VC-relevant knowledge from pitch documents.

Key Features:
- VC-specific ontology (Company, Technology, Market, MarketFriction, Differentiator, ThesisConcept)
- Few-shot prompting with examples
- Iterative refinement capabilities
- Prompt versioning and A/B testing
- Performance monitoring and optimization

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Container for a prompt template with metadata"""
    name: str
    version: str
    description: str
    system_prompt: str
    user_prompt: str
    examples: List[Dict[str, Any]]
    created_at: str
    performance_metrics: Dict[str, float]

@dataclass
class ExtractionExample:
    """Example for few-shot prompting"""
    input_text: str
    expected_output: List[Dict[str, Any]]
    explanation: str

class VCPromptEngineer:
    """
    Prompt engineering system for VC knowledge extraction.
    
    Implements the framework's prompt engineering recommendations:
    - Bespoke VC ontology definition
    - Few-shot prompting with domain examples
    - Iterative refinement based on performance
    - A/B testing capabilities
    """
    
    def __init__(self):
        """Initialize the prompt engineering system"""
        self.ontology = self._define_vc_ontology()
        self.prompt_templates = {}
        self.golden_examples = []
        self.performance_history = []
        
    def _define_vc_ontology(self) -> Dict[str, Dict[str, Any]]:
        """
        Define the bespoke VC ontology from Section 2.3 of the framework.
        
        Returns:
            Dictionary defining entity types and relationship types
        """
        return {
            "entity_types": {
                "Company": {
                    "description": "The startup/firm being analyzed",
                    "required_properties": ["name", "sector"],
                    "optional_properties": ["stage", "funding_amount", "team_size"]
                },
                "Technology": {
                    "description": "Technical approaches, algorithms, platforms, tools",
                    "required_properties": ["name", "category"],
                    "optional_properties": ["maturity_level", "patent_status"]
                },
                "Market": {
                    "description": "Target markets, customer segments, addressable markets",
                    "required_properties": ["name", "size_estimate"],
                    "optional_properties": ["growth_rate", "competitive_intensity"]
                },
                "MarketFriction": {
                    "description": "Problems, pain points, market gaps, inefficiencies",
                    "required_properties": ["description", "severity"],
                    "optional_properties": ["frequency", "existing_solutions"]
                },
                "Differentiator": {
                    "description": "Unique value propositions, competitive advantages, moats",
                    "required_properties": ["description", "type"],
                    "optional_properties": ["sustainability", "time_to_replicate"]
                },
                "ThesisConcept": {
                    "description": "High-level investment theses, strategic assumptions, market beliefs",
                    "required_properties": ["concept", "conviction_level"],
                    "optional_properties": ["time_horizon", "risk_factors"]
                }
            },
            "relationship_types": {
                "has_thesis": "Company -> ThesisConcept",
                "addresses_friction": "Company -> MarketFriction", 
                "has_differentiator": "Company -> Differentiator",
                "targets_market": "Company -> Market",
                "uses_technology": "Company -> Technology",
                "competes_with": "Company -> Company",
                "solves_problem": "Technology -> MarketFriction",
                "enables": "Technology -> Technology",
                "serves": "Technology -> Market",
                "disrupts": "Technology -> Market",
                "depends_on": "Market -> Technology",
                "creates_friction": "Market -> MarketFriction"
            }
        }
    
    def create_extraction_prompt(self, version: str = "v1.0") -> PromptTemplate:
        """
        Create the main knowledge extraction prompt following Section 2.3 guidelines.
        
        Args:
            version: Version identifier for the prompt
            
        Returns:
            PromptTemplate object with the complete extraction prompt
        """
        
        # System prompt - sets the role and context
        system_prompt = """You are a specialized venture capital analyst with expertise in extracting structured investment intelligence from pitch documents. Your task is to analyze text and extract knowledge in the form of Subject-Predicate-Object (SPO) triplets that follow a specific venture capital ontology.

You have deep domain knowledge in:
- Technology assessment and competitive landscapes
- Market analysis and sizing
- Investment thesis evaluation
- Risk assessment and due diligence

Your extractions must be precise, factual, and investment-relevant."""

        # User prompt with detailed instructions
        user_prompt = f"""Extract knowledge triplets from the following text using the VC ontology. Follow these rules precisely:

ONTOLOGY DEFINITIONS:
{self._format_ontology_for_prompt()}

EXTRACTION RULES:
1. PREDICATE CONSTRAINTS: Predicates must be 1-3 words maximum (e.g., "has_thesis", "addresses_friction", "uses_technology")

2. PRONOUN RESOLUTION: Replace all pronouns with the actual entities they refer to

3. ENTITY STANDARDIZATION: Use consistent naming (e.g., "CrowdStrike" not "Crowdstrike" or "CRWD")

4. OUTPUT FORMAT: Return a valid JSON array of objects with this exact structure:
   {{"subject": "entity_name", "predicate": "relationship", "object": "target_entity", "subject_type": "EntityType", "object_type": "EntityType", "confidence": 0.0-1.0}}

5. CONFIDENCE SCORING: Assign confidence scores:
   - 0.9-1.0: Explicitly stated facts
   - 0.7-0.9: Strongly implied information  
   - 0.5-0.7: Reasonably inferred connections
   - Below 0.5: Do not include

VC-SPECIFIC EXTRACTION RULES:

Rule 6 - INVESTMENT THESIS: Identify the core investment thesis or central hypothesis. Model as:
{{"subject": "company_name", "predicate": "has_thesis", "object": "thesis_concept", "subject_type": "Company", "object_type": "ThesisConcept", "confidence": X}}

Rule 7 - MARKET FRICTION: Extract problems, pain points, or market gaps the company addresses:
{{"subject": "company_name", "predicate": "addresses_friction", "object": "specific_problem", "subject_type": "Company", "object_type": "MarketFriction", "confidence": X}}

Rule 8 - DIFFERENTIATORS: Capture competitive advantages, moats, or unique capabilities:
{{"subject": "company_name", "predicate": "has_differentiator", "object": "unique_advantage", "subject_type": "Company", "object_type": "Differentiator", "confidence": X}}

Rule 9 - COMPETITIVE LANDSCAPE: If competitors are mentioned by name:
{{"subject": "company_name", "predicate": "competes_with", "object": "competitor_name", "subject_type": "Company", "object_type": "Company", "confidence": X}}

Rule 10 - TARGET MARKET: Identify target markets or customer segments:
{{"subject": "company_name", "predicate": "targets_market", "object": "market_segment", "subject_type": "Company", "object_type": "Market", "confidence": X}}

Rule 11 - TECHNOLOGY STACK: Extract technologies, platforms, or technical approaches:
{{"subject": "company_name", "predicate": "uses_technology", "object": "technology_name", "subject_type": "Company", "object_type": "Technology", "confidence": X}}

EXAMPLES:
{self._format_examples_for_prompt()}

Now extract knowledge triplets from this text:

"""

        # Create examples for few-shot prompting
        examples = self._create_few_shot_examples()
        
        # Create the prompt template
        template = PromptTemplate(
            name="vc_extraction_prompt",
            version=version,
            description="Main prompt for extracting VC intelligence using bespoke ontology",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=examples,
            created_at=datetime.now().isoformat(),
            performance_metrics={}
        )
        
        self.prompt_templates[f"extraction_{version}"] = template
        return template
    
    def _format_ontology_for_prompt(self) -> str:
        """Format the ontology definition for inclusion in prompts"""
        ontology_text = []
        
        ontology_text.append("ENTITY TYPES:")
        for entity_type, details in self.ontology["entity_types"].items():
            ontology_text.append(f"- {entity_type}: {details['description']}")
        
        ontology_text.append("\nRELATIONSHIP TYPES:")
        for rel_type, description in self.ontology["relationship_types"].items():
            ontology_text.append(f"- {rel_type}: {description}")
        
        return "\n".join(ontology_text)
    
    def _create_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Create few-shot examples for the prompt"""
        return [
            {
                "input": "Acme Security leverages advanced machine learning algorithms to provide autonomous threat detection for enterprise networks. We believe the future of cybersecurity is fully automated response systems that can react faster than human analysts. Our proprietary AI engine can detect zero-day exploits 10x faster than traditional signature-based systems. We compete directly with CrowdStrike and SentinelOne in the endpoint detection market.",
                "output": [
                    {"subject": "Acme Security", "predicate": "uses_technology", "object": "machine learning algorithms", "subject_type": "Company", "object_type": "Technology", "confidence": 0.95},
                    {"subject": "Acme Security", "predicate": "has_thesis", "object": "future of cybersecurity is fully automated response", "subject_type": "Company", "object_type": "ThesisConcept", "confidence": 0.9},
                    {"subject": "Acme Security", "predicate": "has_differentiator", "object": "proprietary AI engine 10x faster detection", "subject_type": "Company", "object_type": "Differentiator", "confidence": 0.85},
                    {"subject": "Acme Security", "predicate": "competes_with", "object": "CrowdStrike", "subject_type": "Company", "object_type": "Company", "confidence": 0.95},
                    {"subject": "Acme Security", "predicate": "competes_with", "object": "SentinelOne", "subject_type": "Company", "object_type": "Company", "confidence": 0.95},
                    {"subject": "Acme Security", "predicate": "targets_market", "object": "endpoint detection market", "subject_type": "Company", "object_type": "Market", "confidence": 0.9}
                ]
            },
            {
                "input": "DefenseTech AI addresses the critical problem of information overload facing military intelligence analysts. Current systems generate too much data for humans to process effectively, creating dangerous blind spots. Our solution uses natural language processing to automatically prioritize and summarize intelligence reports.",
                "output": [
                    {"subject": "DefenseTech AI", "predicate": "addresses_friction", "object": "information overload in military intelligence", "subject_type": "Company", "object_type": "MarketFriction", "confidence": 0.95},
                    {"subject": "DefenseTech AI", "predicate": "addresses_friction", "object": "human inability to process data volume", "subject_type": "Company", "object_type": "MarketFriction", "confidence": 0.9},
                    {"subject": "DefenseTech AI", "predicate": "uses_technology", "object": "natural language processing", "subject_type": "Company", "object_type": "Technology", "confidence": 0.95},
                    {"subject": "DefenseTech AI", "predicate": "targets_market", "object": "military intelligence analysts", "subject_type": "Company", "object_type": "Market", "confidence": 0.9}
                ]
            }
        ]
    
    def _format_examples_for_prompt(self) -> str:
        """Format few-shot examples for the prompt"""
        example_text = []
        for i, example in enumerate(self._create_few_shot_examples(), 1):
            example_text.append(f"Example {i}:")
            example_text.append(f"Input: {example['input']}")
            example_text.append(f"Output: {json.dumps(example['output'], indent=2)}")
            example_text.append("")
        
        return "\n".join(example_text)
    
    def create_standardization_prompt(self, version: str = "v1.0") -> PromptTemplate:
        """Create prompt for entity standardization"""
        
        system_prompt = """You are a data standardization specialist for venture capital knowledge graphs. Your task is to normalize entity names and resolve aliases to create consistent, canonical representations."""
        
        user_prompt = """Standardize the following entities to create canonical representations. Follow these rules:

1. COMPANY NAMES: Use official company names, remove legal suffixes unless critical for disambiguation
   - "Acme Corp" and "Acme Corporation" -> "Acme"
   - "OpenAI Inc." -> "OpenAI"
   
2. TECHNOLOGY NAMES: Use standard technical terminology
   - "AI", "artificial intelligence", "machine learning" -> Use most specific appropriate term
   - "Kubernetes", "k8s" -> "Kubernetes"
   
3. MARKET SEGMENTS: Use industry-standard categorizations
   - "enterprise security", "corporate cybersecurity" -> "enterprise cybersecurity"
   
4. REMOVE ARTICLES: Remove "the", "a", "an" from entity names unless they're part of the official name

5. CASE NORMALIZATION: Use proper capitalization for names, technologies, etc.

Entities to standardize:
{entities}

Return JSON format:
[{{"original": "original_name", "canonical": "standardized_name", "type": "entity_type"}}]
"""
        
        template = PromptTemplate(
            name="standardization_prompt",
            version=version,
            description="Prompt for standardizing entity names and resolving aliases",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=[],
            created_at=datetime.now().isoformat(),
            performance_metrics={}
        )
        
        self.prompt_templates[f"standardization_{version}"] = template
        return template
    
    def create_inference_prompt(self, version: str = "v1.0") -> PromptTemplate:
        """Create prompt for relationship inference"""
        
        system_prompt = """You are a venture capital analyst specializing in identifying implicit relationships between entities in investment contexts. You can infer plausible connections based on domain knowledge and logical reasoning."""
        
        user_prompt = """Given the following entities and explicit relationships, infer additional plausible relationships that are not explicitly stated but can be reasonably deduced from domain knowledge.

INFERENCE RULES:
1. Only infer relationships with confidence >= 0.6
2. Mark all inferred relationships with "inferred": true
3. Base inferences on established VC/tech domain patterns
4. Consider competitive dynamics, technology dependencies, market relationships

EXPLICIT RELATIONSHIPS:
{explicit_relationships}

ENTITIES:
{entities}

Infer additional relationships and return in JSON format:
[{{"subject": "entity1", "predicate": "relationship", "object": "entity2", "subject_type": "Type", "object_type": "Type", "confidence": 0.0-1.0, "inferred": true, "reasoning": "explanation"}}]
"""
        
        template = PromptTemplate(
            name="inference_prompt", 
            version=version,
            description="Prompt for inferring implicit relationships between entities",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=[],
            created_at=datetime.now().isoformat(),
            performance_metrics={}
        )
        
        self.prompt_templates[f"inference_{version}"] = template
        return template
    
    def save_prompt_template(self, template: PromptTemplate, filepath: str):
        """Save a prompt template to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(template), f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved prompt template to {filepath}")
        except Exception as e:
            logger.error(f"Error saving prompt template: {e}")
    
    def load_prompt_template(self, filepath: str) -> Optional[PromptTemplate]:
        """Load a prompt template from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            template = PromptTemplate(**data)
            self.prompt_templates[f"{template.name}_{template.version}"] = template
            logger.info(f"Loaded prompt template from {filepath}")
            return template
        except Exception as e:
            logger.error(f"Error loading prompt template: {e}")
            return None
    
    def evaluate_prompt_performance(self, template_name: str, 
                                  golden_dataset: List[Dict[str, Any]],
                                  extracted_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate prompt performance against golden dataset.
        Implements Section 8.3 evaluation metrics.
        """
        if template_name not in self.prompt_templates:
            logger.error(f"Template {template_name} not found")
            return {}
        
        # Calculate precision, recall, F1
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Convert to sets for comparison
        golden_triplets = set()
        for item in golden_dataset:
            for triplet in item.get('triplets', []):
                key = (triplet['subject'], triplet['predicate'], triplet['object'])
                golden_triplets.add(key)
        
        extracted_triplets = set()
        for triplet in extracted_results:
            key = (triplet['subject'], triplet['predicate'], triplet['object'])
            extracted_triplets.add(key)
        
        # Calculate metrics
        true_positives = len(golden_triplets.intersection(extracted_triplets))
        false_positives = len(extracted_triplets - golden_triplets)
        false_negatives = len(golden_triplets - extracted_triplets)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        # Update template performance metrics
        self.prompt_templates[template_name].performance_metrics.update(metrics)
        
        # Record in performance history
        self.performance_history.append({
            'template_name': template_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        logger.info(f"Prompt performance - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        return metrics
    
    def generate_prompt_variants(self, base_template: PromptTemplate, 
                               variant_configs: List[Dict[str, Any]]) -> List[PromptTemplate]:
        """Generate prompt variants for A/B testing"""
        variants = []
        
        for i, config in enumerate(variant_configs):
            variant_name = f"{base_template.name}_variant_{i+1}"
            variant_version = f"{base_template.version}_v{i+1}"
            
            # Apply modifications based on config
            modified_prompt = base_template.user_prompt
            
            if 'rule_modifications' in config:
                for rule_mod in config['rule_modifications']:
                    if rule_mod['action'] == 'add':
                        modified_prompt += f"\n\nAdditional Rule: {rule_mod['content']}"
                    elif rule_mod['action'] == 'replace':
                        modified_prompt = modified_prompt.replace(rule_mod['target'], rule_mod['content'])
            
            if 'confidence_threshold' in config:
                threshold = config['confidence_threshold']
                modified_prompt = modified_prompt.replace(
                    "Below 0.5: Do not include",
                    f"Below {threshold}: Do not include"
                )
            
            variant = PromptTemplate(
                name=variant_name,
                version=variant_version,
                description=f"Variant of {base_template.name}: {config.get('description', 'Modified version')}",
                system_prompt=base_template.system_prompt,
                user_prompt=modified_prompt,
                examples=base_template.examples,
                created_at=datetime.now().isoformat(),
                performance_metrics={}
            )
            
            variants.append(variant)
            self.prompt_templates[f"{variant_name}_{variant_version}"] = variant
        
        logger.info(f"Generated {len(variants)} prompt variants")
        return variants
    
    def get_best_performing_prompt(self, task_type: str = "extraction") -> Optional[PromptTemplate]:
        """Get the best performing prompt template for a given task"""
        candidates = [t for name, t in self.prompt_templates.items() if task_type in name.lower()]
        
        if not candidates:
            return None
        
        # Sort by F1 score
        candidates_with_scores = []
        for template in candidates:
            f1_score = template.performance_metrics.get('f1_score', 0)
            candidates_with_scores.append((template, f1_score))
        
        if not candidates_with_scores:
            return candidates[0]  # Return first if no performance data
        
        best_template = max(candidates_with_scores, key=lambda x: x[1])[0]
        logger.info(f"Best performing {task_type} template: {best_template.name} (F1: {best_template.performance_metrics.get('f1_score', 'N/A')})")
        return best_template

def main():
    """Demo of the prompt engineering system"""
    engineer = VCPromptEngineer()
    
    # Create main extraction prompt
    extraction_prompt = engineer.create_extraction_prompt("v1.0")
    print("=== EXTRACTION PROMPT ===")
    print(f"System: {extraction_prompt.system_prompt[:200]}...")
    print(f"User: {extraction_prompt.user_prompt[:500]}...")
    
    # Create variants for A/B testing
    variant_configs = [
        {
            'description': 'Lower confidence threshold',
            'confidence_threshold': 0.4
        },
        {
            'description': 'Additional market sizing rule',
            'rule_modifications': [{
                'action': 'add',
                'content': 'Rule 12 - MARKET SIZE: Extract market size estimates: {"subject": "market_name", "predicate": "has_size", "object": "size_estimate", "subject_type": "Market", "object_type": "MarketMetric", "confidence": X}'
            }]
        }
    ]
    
    variants = engineer.generate_prompt_variants(extraction_prompt, variant_configs)
    print(f"\nGenerated {len(variants)} prompt variants for A/B testing")
    
    # Save templates
    engineer.save_prompt_template(extraction_prompt, "extraction_prompt_v1.0.yaml")
    for variant in variants:
        engineer.save_prompt_template(variant, f"{variant.name}_{variant.version}.yaml")
    
    print("\nPrompt templates saved successfully!")

if __name__ == "__main__":
    main() 