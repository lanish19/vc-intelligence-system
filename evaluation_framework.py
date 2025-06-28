#!/usr/bin/env python3
"""
Knowledge Extraction Evaluation Framework

This module implements Section 8 of the strategic framework: "A Framework for Evaluating 
and Enhancing Extraction Quality". It provides comprehensive evaluation capabilities for 
the knowledge graph extraction system including golden dataset creation, performance 
metrics, and iterative improvement workflows.

Key Features:
- Golden dataset management
- Classical IR metrics (Precision, Recall, F1)
- LLM-as-a-judge evaluation
- Performance tracking and comparison
- Error analysis and improvement suggestions

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from datetime import datetime
import hashlib
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    total_predictions: int
    total_ground_truth: int

@dataclass
class ErrorAnalysis:
    """Container for error analysis results"""
    missed_entities: List[str]
    missed_relationships: List[str]
    false_entities: List[str]
    false_relationships: List[str]
    common_error_patterns: Dict[str, int]
    improvement_suggestions: List[str]

@dataclass
class GoldenDatasetItem:
    """Single item in the golden dataset"""
    document_id: str
    source_text: str
    ground_truth_triplets: List[Dict[str, Any]]
    document_type: str
    sector: str
    created_by: str
    created_at: str
    validation_notes: str

class EvaluationFramework:
    """
    Comprehensive evaluation system for knowledge extraction quality.
    
    Implements Section 8 framework:
    - Golden dataset creation and management
    - Multiple evaluation metrics
    - Error analysis and improvement tracking
    - Performance comparison across versions
    """
    
    def __init__(self, golden_dataset_path: Optional[str] = None):
        """
        Initialize the evaluation framework.
        
        Args:
            golden_dataset_path: Path to existing golden dataset file
        """
        self.golden_dataset = []
        self.evaluation_history = []
        self.performance_tracking = defaultdict(list)
        
        if golden_dataset_path and Path(golden_dataset_path).exists():
            self.load_golden_dataset(golden_dataset_path)
    
    def create_golden_dataset_item(self, document_id: str, source_text: str, 
                                 document_type: str = "pitch_deck", 
                                 sector: str = "unknown") -> GoldenDatasetItem:
        """
        Create a new golden dataset item for manual annotation.
        
        Args:
            document_id: Unique identifier for the document
            source_text: The source text to be annotated
            document_type: Type of document (e.g., "pitch_deck", "executive_summary")
            sector: Business sector (e.g., "cybersecurity", "autonomous_systems")
            
        Returns:
            GoldenDatasetItem ready for annotation
        """
        return GoldenDatasetItem(
            document_id=document_id,
            source_text=source_text,
            ground_truth_triplets=[],  # To be filled by human annotator
            document_type=document_type,
            sector=sector,
            created_by="",  # To be filled by annotator
            created_at=datetime.now().isoformat(),
            validation_notes=""
        )
    
    def add_to_golden_dataset(self, item: GoldenDatasetItem):
        """Add an annotated item to the golden dataset"""
        # Validate the item
        if not item.ground_truth_triplets:
            logger.warning(f"Adding item {item.document_id} with no ground truth triplets")
        
        # Check for duplicates
        existing_ids = [existing.document_id for existing in self.golden_dataset]
        if item.document_id in existing_ids:
            logger.warning(f"Document {item.document_id} already exists in golden dataset")
            return
        
        self.golden_dataset.append(item)
        logger.info(f"Added document {item.document_id} to golden dataset (total: {len(self.golden_dataset)})")
    
    def save_golden_dataset(self, filepath: str):
        """Save the golden dataset to file"""
        try:
            golden_data = [asdict(item) for item in self.golden_dataset]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(golden_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved golden dataset with {len(self.golden_dataset)} items to {filepath}")
        except Exception as e:
            logger.error(f"Error saving golden dataset: {e}")
    
    def load_golden_dataset(self, filepath: str) -> bool:
        """Load golden dataset from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                golden_data = json.load(f)
            
            self.golden_dataset = []
            for item_data in golden_data:
                item = GoldenDatasetItem(**item_data)
                self.golden_dataset.append(item)
            
            logger.info(f"Loaded golden dataset with {len(self.golden_dataset)} items")
            return True
            
        except Exception as e:
            logger.error(f"Error loading golden dataset: {e}")
            return False
    
    def calculate_classical_metrics(self, predicted_triplets: List[Dict[str, Any]], 
                                  ground_truth_triplets: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Calculate classical information retrieval metrics.
        
        Implements Section 8.3 evaluation metrics from the framework.
        
        Args:
            predicted_triplets: Triplets extracted by the system
            ground_truth_triplets: Manually annotated ground truth triplets
            
        Returns:
            EvaluationMetrics object with calculated scores
        """
        
        # Convert triplets to comparable format (ignoring confidence scores, etc.)
        def normalize_triplet(triplet):
            return (
                triplet['subject'].lower().strip(),
                triplet['predicate'].lower().strip(), 
                triplet['object'].lower().strip()
            )
        
        predicted_set = set(normalize_triplet(t) for t in predicted_triplets)
        ground_truth_set = set(normalize_triplet(t) for t in ground_truth_triplets)
        
        # Calculate metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        # Classical metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        total_predictions = len(predicted_set)
        total_ground_truth = len(ground_truth_set)
        accuracy = true_positives / max(total_predictions, total_ground_truth) if max(total_predictions, total_ground_truth) > 0 else 0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            accuracy=accuracy,
            total_predictions=total_predictions,
            total_ground_truth=total_ground_truth
        )
    
    def perform_error_analysis(self, predicted_triplets: List[Dict[str, Any]], 
                             ground_truth_triplets: List[Dict[str, Any]]) -> ErrorAnalysis:
        """
        Perform detailed error analysis to identify improvement opportunities.
        
        Args:
            predicted_triplets: System predictions
            ground_truth_triplets: Ground truth annotations
            
        Returns:
            ErrorAnalysis object with detailed error breakdown
        """
        
        def normalize_triplet(triplet):
            return (triplet['subject'].lower().strip(), triplet['predicate'].lower().strip(), triplet['object'].lower().strip())
        
        predicted_set = set(normalize_triplet(t) for t in predicted_triplets)
        ground_truth_set = set(normalize_triplet(t) for t in ground_truth_triplets)
        
        # Identify different types of errors
        missed_triplets = ground_truth_set - predicted_set  # False negatives
        false_triplets = predicted_set - ground_truth_set   # False positives
        
        # Analyze entity-level errors
        predicted_entities = set()
        ground_truth_entities = set()
        
        for triplet in predicted_triplets:
            predicted_entities.add(triplet['subject'].lower().strip())
            predicted_entities.add(triplet['object'].lower().strip())
        
        for triplet in ground_truth_triplets:
            ground_truth_entities.add(triplet['subject'].lower().strip())
            ground_truth_entities.add(triplet['object'].lower().strip())
        
        missed_entities = list(ground_truth_entities - predicted_entities)
        false_entities = list(predicted_entities - ground_truth_entities)
        
        # Analyze relationship-level errors
        predicted_relationships = set(normalize_triplet(t)[1] for t in predicted_triplets)
        ground_truth_relationships = set(normalize_triplet(t)[1] for t in ground_truth_triplets)
        
        missed_relationships = list(ground_truth_relationships - predicted_relationships)
        false_relationships = list(predicted_relationships - ground_truth_relationships)
        
        # Identify common error patterns
        error_patterns = Counter()
        
        for missed in missed_triplets:
            subject, predicate, obj = missed
            # Pattern: missing specific relationship types
            error_patterns[f"missed_relation_{predicate}"] += 1
            # Pattern: missing specific entity types
            error_patterns[f"missed_entity_in_subject"] += 1
            error_patterns[f"missed_entity_in_object"] += 1
        
        for false_triplet in false_triplets:
            subject, predicate, obj = false_triplet
            error_patterns[f"false_relation_{predicate}"] += 1
            error_patterns[f"false_entity_extraction"] += 1
        
        # Generate improvement suggestions
        suggestions = []
        
        if len(missed_entities) > 5:
            suggestions.append(f"Entity extraction needs improvement: {len(missed_entities)} entities missed")
        
        if len(false_entities) > 5:
            suggestions.append(f"Entity extraction too liberal: {len(false_entities)} false entities")
        
        common_missed_relations = [rel for rel in missed_relationships if missed_relationships.count(rel) > 1]
        if common_missed_relations:
            suggestions.append(f"Improve detection of relationships: {', '.join(common_missed_relations)}")
        
        if error_patterns.get('missed_relation_has_thesis', 0) > 2:
            suggestions.append("Add more examples for investment thesis extraction")
        
        if error_patterns.get('missed_relation_addresses_friction', 0) > 2:
            suggestions.append("Improve market friction detection with better patterns")
        
        return ErrorAnalysis(
            missed_entities=missed_entities[:10],  # Limit for readability
            missed_relationships=missed_relationships,
            false_entities=false_entities[:10],
            false_relationships=false_relationships,
            common_error_patterns=dict(error_patterns.most_common(10)),
            improvement_suggestions=suggestions
        )
    
    def evaluate_system(self, system_predictions: Dict[str, List[Dict[str, Any]]], 
                       system_version: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate the extraction system against the golden dataset.
        
        Args:
            system_predictions: Dictionary mapping document_id to extracted triplets
            system_version: Version identifier for tracking
            
        Returns:
            Comprehensive evaluation results
        """
        if not self.golden_dataset:
            logger.error("No golden dataset available for evaluation")
            return {}
        
        logger.info(f"Evaluating system version '{system_version}' against {len(self.golden_dataset)} documents")
        
        # Collect all metrics and errors
        all_metrics = []
        all_errors = []
        document_results = {}
        
        for golden_item in self.golden_dataset:
            doc_id = golden_item.document_id
            
            if doc_id not in system_predictions:
                logger.warning(f"No predictions found for document {doc_id}")
                continue
            
            predicted_triplets = system_predictions[doc_id]
            ground_truth_triplets = golden_item.ground_truth_triplets
            
            # Calculate metrics for this document
            metrics = self.calculate_classical_metrics(predicted_triplets, ground_truth_triplets)
            errors = self.perform_error_analysis(predicted_triplets, ground_truth_triplets)
            
            all_metrics.append(metrics)
            all_errors.append(errors)
            
            document_results[doc_id] = {
                'metrics': asdict(metrics),
                'errors': asdict(errors),
                'sector': golden_item.sector,
                'document_type': golden_item.document_type
            }
        
        if not all_metrics:
            logger.error("No valid evaluations could be performed")
            return {}
        
        # Calculate aggregate metrics
        aggregate_metrics = EvaluationMetrics(
            precision=np.mean([m.precision for m in all_metrics]),
            recall=np.mean([m.recall for m in all_metrics]),
            f1_score=np.mean([m.f1_score for m in all_metrics]),
            true_positives=sum(m.true_positives for m in all_metrics),
            false_positives=sum(m.false_positives for m in all_metrics),
            false_negatives=sum(m.false_negatives for m in all_metrics),
            accuracy=np.mean([m.accuracy for m in all_metrics]),
            total_predictions=sum(m.total_predictions for m in all_metrics),
            total_ground_truth=sum(m.total_ground_truth for m in all_metrics)
        )
        
        # Aggregate error analysis
        all_missed_entities = []
        all_false_entities = []
        all_suggestions = []
        for error in all_errors:
            all_missed_entities.extend(error.missed_entities)
            all_false_entities.extend(error.false_entities)
            all_suggestions.extend(error.improvement_suggestions)
        
        aggregate_errors = ErrorAnalysis(
            missed_entities=list(set(all_missed_entities)),
            missed_relationships=list(set([rel for error in all_errors for rel in error.missed_relationships])),
            false_entities=list(set(all_false_entities)),
            false_relationships=list(set([rel for error in all_errors for rel in error.false_relationships])),
            common_error_patterns={},
            improvement_suggestions=list(set(all_suggestions))
        )
        
        # Compile results
        evaluation_results = {
            'system_version': system_version,
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_size': len(self.golden_dataset),
            'documents_evaluated': len(document_results),
            'aggregate_metrics': asdict(aggregate_metrics),
            'aggregate_errors': asdict(aggregate_errors),
            'document_results': document_results,
            'performance_summary': {
                'precision_mean': aggregate_metrics.precision,
                'precision_std': np.std([m.precision for m in all_metrics]),
                'recall_mean': aggregate_metrics.recall,
                'recall_std': np.std([m.recall for m in all_metrics]),
                'f1_mean': aggregate_metrics.f1_score,
                'f1_std': np.std([m.f1_score for m in all_metrics])
            }
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_results)
        self.performance_tracking[system_version].append(aggregate_metrics.f1_score)
        
        # Log summary
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"System Version: {system_version}")
        logger.info(f"Documents Evaluated: {len(document_results)}/{len(self.golden_dataset)}")
        logger.info(f"Precision: {aggregate_metrics.precision:.3f}")
        logger.info(f"Recall: {aggregate_metrics.recall:.3f}")
        logger.info(f"F1-Score: {aggregate_metrics.f1_score:.3f}")
        logger.info(f"True Positives: {aggregate_metrics.true_positives}")
        logger.info(f"False Positives: {aggregate_metrics.false_positives}")
        logger.info(f"False Negatives: {aggregate_metrics.false_negatives}")
        
        if aggregate_errors.improvement_suggestions:
            logger.info("Top Improvement Suggestions:")
            for i, suggestion in enumerate(aggregate_errors.improvement_suggestions[:5], 1):
                logger.info(f"  {i}. {suggestion}")
        
        return evaluation_results
    
    def compare_system_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare performance between two system versions"""
        
        v1_results = [r for r in self.evaluation_history if r['system_version'] == version1]
        v2_results = [r for r in self.evaluation_history if r['system_version'] == version2]
        
        if not v1_results or not v2_results:
            logger.error(f"Missing evaluation results for comparison")
            return {}
        
        v1_latest = v1_results[-1]['aggregate_metrics']
        v2_latest = v2_results[-1]['aggregate_metrics']
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'precision_change': v2_latest['precision'] - v1_latest['precision'],
            'recall_change': v2_latest['recall'] - v1_latest['recall'],
            'f1_change': v2_latest['f1_score'] - v1_latest['f1_score'],
            'better_version': version2 if v2_latest['f1_score'] > v1_latest['f1_score'] else version1
        }
        
        logger.info(f"Version Comparison: {version1} vs {version2}")
        logger.info(f"F1 Change: {comparison['f1_change']:+.3f}")
        logger.info(f"Better Version: {comparison['better_version']}")
        
        return comparison
    
    def generate_evaluation_report(self, output_path: str, system_version: str = None):
        """Generate a comprehensive evaluation report"""
        
        if not self.evaluation_history:
            logger.error("No evaluation history available")
            return
        
        # Use latest evaluation if no version specified
        if system_version:
            results = [r for r in self.evaluation_history if r['system_version'] == system_version]
            if not results:
                logger.error(f"No results found for version {system_version}")
                return
            latest_results = results[-1]
        else:
            latest_results = self.evaluation_history[-1]
        
        # Generate report
        report = {
            'report_generated': datetime.now().isoformat(),
            'evaluation_framework_version': '1.0',
            'latest_evaluation': latest_results,
            'performance_trends': {
                version: scores for version, scores in self.performance_tracking.items()
            },
            'golden_dataset_summary': {
                'total_documents': len(self.golden_dataset),
                'sectors': list(set(item.sector for item in self.golden_dataset)),
                'document_types': list(set(item.document_type for item in self.golden_dataset))
            }
        }
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

def main():
    """Demo of the evaluation framework"""
    framework = EvaluationFramework()
    
    # Create sample golden dataset item
    sample_text = """Acme Security uses machine learning for threat detection. 
    We compete with CrowdStrike and address the problem of slow incident response."""
    
    golden_item = framework.create_golden_dataset_item(
        document_id="demo_doc_1",
        source_text=sample_text,
        sector="cybersecurity"
    )
    
    # Add ground truth (normally done by human annotator)
    golden_item.ground_truth_triplets = [
        {"subject": "Acme Security", "predicate": "uses_technology", "object": "machine learning", "subject_type": "Company", "object_type": "Technology"},
        {"subject": "Acme Security", "predicate": "competes_with", "object": "CrowdStrike", "subject_type": "Company", "object_type": "Company"},
        {"subject": "Acme Security", "predicate": "addresses_friction", "object": "slow incident response", "subject_type": "Company", "object_type": "MarketFriction"}
    ]
    golden_item.created_by = "demo_user"
    
    framework.add_to_golden_dataset(golden_item)
    
    # Simulate system predictions
    system_predictions = {
        "demo_doc_1": [
            {"subject": "Acme Security", "predicate": "uses_technology", "object": "machine learning", "subject_type": "Company", "object_type": "Technology"},
            {"subject": "Acme Security", "predicate": "competes_with", "object": "CrowdStrike", "subject_type": "Company", "object_type": "Company"},
            # Missing one correct triplet, adding one incorrect
            {"subject": "Acme Security", "predicate": "targets_market", "object": "enterprise", "subject_type": "Company", "object_type": "Market"}
        ]
    }
    
    # Evaluate system
    results = framework.evaluate_system(system_predictions, "demo_v1.0")
    
    # Save results
    framework.save_golden_dataset("demo_golden_dataset.json")
    framework.generate_evaluation_report("demo_evaluation_report.json")
    
    print("Demo evaluation completed successfully!")

if __name__ == "__main__":
    main() 