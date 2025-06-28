#!/usr/bin/env python3
"""
Knowledge Extraction Evaluation System

This module implements Section 8 of the strategic framework for evaluating 
and enhancing extraction quality with golden datasets and performance metrics.

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Evaluation metrics container"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int

class EvaluationSystem:
    """
    Evaluation system for knowledge extraction quality assessment.
    
    Implements Section 8 framework requirements:
    - Golden dataset management
    - Classical IR metrics calculation
    - Performance tracking
    - Error analysis
    """
    
    def __init__(self):
        self.golden_dataset = []
        self.evaluation_history = []
    
    def add_golden_example(self, document_id: str, source_text: str, 
                          ground_truth_triplets: List[Dict[str, Any]]):
        """Add a manually annotated example to the golden dataset"""
        golden_item = {
            'document_id': document_id,
            'source_text': source_text,
            'ground_truth_triplets': ground_truth_triplets,
            'created_at': datetime.now().isoformat()
        }
        self.golden_dataset.append(golden_item)
        logger.info(f"Added golden example {document_id} (total: {len(self.golden_dataset)})")
    
    def calculate_metrics(self, predicted: List[Dict[str, Any]], 
                         ground_truth: List[Dict[str, Any]]) -> EvaluationMetrics:
        """
        Calculate precision, recall, and F1-score.
        
        Implements Section 8.3 classical metrics from the framework.
        """
        
        # Normalize triplets for comparison
        def normalize_triplet(triplet):
            return (
                triplet['subject'].lower().strip(),
                triplet['predicate'].lower().strip(),
                triplet['object'].lower().strip()
            )
        
        predicted_set = set(normalize_triplet(t) for t in predicted)
        ground_truth_set = set(normalize_triplet(t) for t in ground_truth)
        
        # Calculate metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def evaluate_system(self, system_predictions: Dict[str, List[Dict[str, Any]]], 
                       version: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate the extraction system against the golden dataset.
        
        Args:
            system_predictions: Map of document_id -> extracted triplets
            version: System version identifier
            
        Returns:
            Evaluation results including metrics and analysis
        """
        if not self.golden_dataset:
            logger.error("No golden dataset available")
            return {}
        
        all_metrics = []
        document_results = {}
        
        for golden_item in self.golden_dataset:
            doc_id = golden_item['document_id']
            
            if doc_id not in system_predictions:
                logger.warning(f"No predictions for document {doc_id}")
                continue
            
            predicted = system_predictions[doc_id]
            ground_truth = golden_item['ground_truth_triplets']
            
            metrics = self.calculate_metrics(predicted, ground_truth)
            all_metrics.append(metrics)
            
            document_results[doc_id] = {
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives,
                'false_negatives': metrics.false_negatives
            }
        
        # Calculate aggregate metrics
        if all_metrics:
            aggregate_precision = np.mean([m.precision for m in all_metrics])
            aggregate_recall = np.mean([m.recall for m in all_metrics])
            aggregate_f1 = np.mean([m.f1_score for m in all_metrics])
            total_tp = sum(m.true_positives for m in all_metrics)
            total_fp = sum(m.false_positives for m in all_metrics)
            total_fn = sum(m.false_negatives for m in all_metrics)
        else:
            aggregate_precision = aggregate_recall = aggregate_f1 = 0
            total_tp = total_fp = total_fn = 0
        
        results = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(self.golden_dataset),
            'documents_evaluated': len(document_results),
            'aggregate_metrics': {
                'precision': aggregate_precision,
                'recall': aggregate_recall,
                'f1_score': aggregate_f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            },
            'document_results': document_results
        }
        
        self.evaluation_history.append(results)
        
        logger.info(f"=== EVALUATION RESULTS ({version}) ===")
        logger.info(f"Documents evaluated: {len(document_results)}")
        logger.info(f"Precision: {aggregate_precision:.3f}")
        logger.info(f"Recall: {aggregate_recall:.3f}")
        logger.info(f"F1-Score: {aggregate_f1:.3f}")
        
        return results
    
    def save_golden_dataset(self, filepath: str):
        """Save golden dataset to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.golden_dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved golden dataset to {filepath}")
        except Exception as e:
            logger.error(f"Error saving golden dataset: {e}")
    
    def load_golden_dataset(self, filepath: str):
        """Load golden dataset from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.golden_dataset = json.load(f)
            logger.info(f"Loaded golden dataset with {len(self.golden_dataset)} items")
        except Exception as e:
            logger.error(f"Error loading golden dataset: {e}")
    
    def generate_improvement_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on evaluation results"""
        suggestions = []
        metrics = results['aggregate_metrics']
        
        if metrics['precision'] < 0.7:
            suggestions.append("Low precision: Add more specific extraction rules to reduce false positives")
        
        if metrics['recall'] < 0.7:
            suggestions.append("Low recall: Add more pattern matching to capture missed entities")
        
        if metrics['false_negatives'] > metrics['true_positives']:
            suggestions.append("High false negative rate: Review prompt examples and add more training cases")
        
        if metrics['false_positives'] > metrics['true_positives']:
            suggestions.append("High false positive rate: Increase confidence thresholds or add filtering")
        
        return suggestions

def main():
    """Demo the evaluation system"""
    evaluator = EvaluationSystem()
    
    # Create sample golden dataset
    evaluator.add_golden_example(
        document_id="demo_1",
        source_text="Acme Security uses AI for threat detection and competes with CrowdStrike",
        ground_truth_triplets=[
            {"subject": "Acme Security", "predicate": "uses_technology", "object": "AI", "subject_type": "Company", "object_type": "Technology"},
            {"subject": "Acme Security", "predicate": "competes_with", "object": "CrowdStrike", "subject_type": "Company", "object_type": "Company"}
        ]
    )
    
    # Simulate system predictions
    predictions = {
        "demo_1": [
            {"subject": "Acme Security", "predicate": "uses_technology", "object": "AI", "subject_type": "Company", "object_type": "Technology"},
            # Missing the competition relationship
        ]
    }
    
    # Evaluate
    results = evaluator.evaluate_system(predictions, "demo_v1.0")
    suggestions = evaluator.generate_improvement_suggestions(results)
    
    print("\nImprovement Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")

if __name__ == "__main__":
    main() 