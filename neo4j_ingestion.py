#!/usr/bin/env python3
"""
Neo4j Meta-Graph Ingestion System

This module implements Section 3 of the strategic framework: "Architecting the Meta-Graph 
for Portfolio-Wide Analysis". It creates a unified, queryable knowledge graph from the 
extracted SPO triplets using Neo4j as the persistent graph database.

Key Features:
- Batch processing for performance
- MERGE operations to prevent duplicates 
- Index creation for query optimization
- Data lineage tracking
- Comprehensive error handling and logging

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class IngestionStats:
    """Statistics for ingestion process"""
    total_triplets: int = 0
    successful_ingestions: int = 0
    failed_ingestions: int = 0
    unique_entities: int = 0
    unique_relationships: int = 0
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class Neo4jIngestion:
    """
    Handles ingestion of VC knowledge triplets into Neo4j meta-graph.
    
    Implements the strategic framework's requirements for:
    - Unified entity resolution via MERGE operations
    - Scalable batch processing
    - Data lineage preservation
    - Performance optimization through indexing
    """
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver: Optional[Driver] = None
        self.entity_label = "Entity"
        self.relationship_type = "RELATES_TO"
        
        # Indexes for performance
        self.required_indexes = [
            f"CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:{self.entity_label}) ON (e.name)",
            f"CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:{self.entity_label}) ON (e.type)",
            f"CREATE INDEX entity_composite_index IF NOT EXISTS FOR (e:{self.entity_label}) ON (e.name, e.type)",
        ]
        
        # Constraints for data integrity
        self.required_constraints = [
            f"CREATE CONSTRAINT entity_unique_name IF NOT EXISTS FOR (e:{self.entity_label}) REQUIRE e.name IS UNIQUE"
        ]
    
    def connect(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
            
            # Test connection
            with self.driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info(f"Successfully connected to Neo4j at {self.config.uri}")
                    return True
                    
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            return False
        
        return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    def setup_database(self) -> bool:
        """Set up indexes and constraints for optimal performance"""
        if not self.driver:
            logger.error("No database connection established")
            return False
        
        try:
            with self.driver.session(database=self.config.database) as session:
                
                # Create indexes
                for index_query in self.required_indexes:
                    try:
                        session.run(index_query)
                        logger.info(f"Created/verified index: {index_query.split('(')[0]}")
                    except ClientError as e:
                        logger.warning(f"Index creation warning: {e}")
                
                # Note: Constraints commented out as they might conflict with MERGE operations
                # In production, carefully consider unique constraints vs. entity resolution needs
                
                logger.info("Database setup completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            return False
    
    def ingest_json_file(self, json_path: str, source_document: str = None, 
                        batch_size: int = 1000) -> IngestionStats:
        """
        Ingest triplets from a JSON file into the meta-graph.
        
        Args:
            json_path: Path to JSON file containing triplets
            source_document: Optional source document identifier
            batch_size: Number of triplets to process in each batch
            
        Returns:
            IngestionStats with processing results
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
        
        logger.info(f"Starting ingestion from {json_path}")
        start_time = time.time()
        
        # Load triplets
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                triplets = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {json_path}: {e}")
            return IngestionStats(errors=[f"File loading error: {e}"])
        
        if not isinstance(triplets, list):
            logger.error("JSON file must contain a list of triplets")
            return IngestionStats(errors=["Invalid JSON format: expected list"])
        
        # Source document fallback
        if source_document is None:
            source_document = Path(json_path).stem
        
        # Process in batches
        stats = IngestionStats(total_triplets=len(triplets))
        
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i + batch_size]
            batch_stats = self._ingest_batch(batch, source_document, i // batch_size + 1)
            
            # Aggregate stats
            stats.successful_ingestions += batch_stats.successful_ingestions
            stats.failed_ingestions += batch_stats.failed_ingestions
            stats.errors.extend(batch_stats.errors)
            
            logger.info(f"Processed batch {i // batch_size + 1}, "
                       f"success: {batch_stats.successful_ingestions}, "
                       f"failed: {batch_stats.failed_ingestions}")
        
        # Calculate final stats
        stats.processing_time = time.time() - start_time
        stats.unique_entities, stats.unique_relationships = self._get_graph_stats()
        
        logger.info(f"Ingestion completed in {stats.processing_time:.2f}s. "
                   f"Success: {stats.successful_ingestions}, Failed: {stats.failed_ingestions}")
        
        return stats
    
    def _ingest_batch(self, batch: List[Dict], source_document: str, 
                     batch_number: int) -> IngestionStats:
        """Process a batch of triplets"""
        stats = IngestionStats()
        
        with self.driver.session(database=self.config.database) as session:
            for triplet in batch:
                try:
                    self._ingest_single_triplet(session, triplet, source_document)
                    stats.successful_ingestions += 1
                except Exception as e:
                    stats.failed_ingestions += 1
                    error_msg = f"Batch {batch_number}, triplet error: {e}"
                    stats.errors.append(error_msg)
                    logger.warning(error_msg)
        
        return stats
    
    def _ingest_single_triplet(self, session: Session, triplet: Dict, 
                              source_document: str):
        """
        Ingest a single triplet using MERGE for entity resolution.
        
        This is the core method implementing the strategic framework's
        requirement for unified entity resolution across the portfolio.
        """
        
        # Validate triplet structure
        required_fields = {'subject', 'predicate', 'object', 'subject_type', 'object_type'}
        if not all(field in triplet for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields - triplet.keys()}")
        
        # Enhanced Cypher query with comprehensive MERGE logic
        cypher_query = """
        // MERGE subject entity with type and properties
        MERGE (subject:Entity {name: $subject_name})
        ON CREATE SET 
            subject.type = $subject_type,
            subject.created_at = datetime(),
            subject.first_seen_in = $source_document
        ON MATCH SET
            subject.last_updated = datetime(),
            subject.seen_in = coalesce(subject.seen_in, []) + $source_document
        
        // MERGE object entity with type and properties  
        MERGE (object:Entity {name: $object_name})
        ON CREATE SET
            object.type = $object_type,
            object.created_at = datetime(),
            object.first_seen_in = $source_document
        ON MATCH SET
            object.last_updated = datetime(),
            object.seen_in = coalesce(object.seen_in, []) + $source_document
        
        // MERGE relationship with full metadata
        MERGE (subject)-[rel:RELATES_TO {
            predicate: $predicate,
            subject_name: $subject_name,
            object_name: $object_name
        }]->(object)
        ON CREATE SET
            rel.confidence = $confidence,
            rel.source_document = $source_document,
            rel.source_field = $source_field,
            rel.source_row = $source_row,
            rel.extracted_at = $extracted_at,
            rel.created_at = datetime(),
            rel.relationship_id = randomUUID()
        ON MATCH SET
            rel.last_seen = datetime(),
            rel.occurrence_count = coalesce(rel.occurrence_count, 1) + 1
        
        RETURN subject.name, object.name, rel.predicate
        """
        
        # Prepare parameters
        parameters = {
            'subject_name': str(triplet['subject']).strip(),
            'object_name': str(triplet['object']).strip(), 
            'predicate': str(triplet['predicate']).strip(),
            'subject_type': str(triplet['subject_type']).strip(),
            'object_type': str(triplet['object_type']).strip(),
            'confidence': float(triplet.get('confidence', 0.8)),
            'source_document': source_document,
            'source_field': str(triplet.get('source_field', 'unknown')),
            'source_row': int(triplet.get('source_row', -1)),
            'extracted_at': str(triplet.get('extracted_at', datetime.now().isoformat()))
        }
        
        # Execute query
        result = session.run(cypher_query, parameters)
        
        # Consume result to ensure completion
        summary = result.consume()
        
        # Log any warnings
        if summary.notifications:
            for notification in summary.notifications:
                logger.warning(f"Neo4j notification: {notification.description}")
    
    def ingest_multiple_files(self, file_paths: List[str], 
                             batch_size: int = 1000) -> IngestionStats:
        """Ingest multiple JSON files into the meta-graph"""
        
        combined_stats = IngestionStats()
        
        for file_path in file_paths:
            logger.info(f"Processing file {file_path}")
            file_stats = self.ingest_json_file(file_path, batch_size=batch_size)
            
            # Combine statistics
            combined_stats.total_triplets += file_stats.total_triplets
            combined_stats.successful_ingestions += file_stats.successful_ingestions
            combined_stats.failed_ingestions += file_stats.failed_ingestions
            combined_stats.processing_time += file_stats.processing_time
            combined_stats.errors.extend(file_stats.errors)
        
        # Get final graph stats
        combined_stats.unique_entities, combined_stats.unique_relationships = self._get_graph_stats()
        
        return combined_stats
    
    def _get_graph_stats(self) -> tuple[int, int]:
        """Get current graph statistics"""
        if not self.driver:
            return 0, 0
        
        try:
            with self.driver.session(database=self.config.database) as session:
                # Count entities
                entity_result = session.run(f"MATCH (e:{self.entity_label}) RETURN count(e) as count")
                entity_count = entity_result.single()["count"]
                
                # Count relationships
                rel_result = session.run(f"MATCH ()-[r:{self.relationship_type}]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
                
                return entity_count, rel_count
                
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return 0, 0
    
    def query_company_subgraph(self, company_name: str, depth: int = 2) -> Dict[str, Any]:
        """
        Extract a company's subgraph for analysis.
        Implements the strategic framework's subgraph comparison capability.
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
        
        cypher_query = f"""
        MATCH (company:{self.entity_label} {{name: $company_name, type: 'Company'}})
        OPTIONAL MATCH path = (company)-[r:{self.relationship_type}*1..{depth}]-(connected)
        RETURN company, 
               collect(DISTINCT connected) as connected_entities,
               collect(DISTINCT r) as relationships,
               collect(DISTINCT path) as paths
        """
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher_query, company_name=company_name)
            record = result.single()
            
            if not record or not record["company"]:
                return {"error": f"Company '{company_name}' not found"}
            
            return {
                "company": dict(record["company"]),
                "connected_entities": [dict(entity) for entity in record["connected_entities"] if entity],
                "relationships": [dict(rel) for rel in record["relationships"] if rel],
                "subgraph_size": {
                    "entities": len(record["connected_entities"]) + 1,  # +1 for company itself
                    "relationships": len([r for r in record["relationships"] if r])
                }
            }
    
    def find_shared_connections(self, company1: str, company2: str) -> Dict[str, Any]:
        """
        Find shared connections between two companies.
        Supports the strategic framework's thesis alignment analysis.
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
        
        cypher_query = f"""
        MATCH (c1:{self.entity_label} {{name: $company1, type: 'Company'}})
        MATCH (c2:{self.entity_label} {{name: $company2, type: 'Company'}})
        MATCH (c1)-[r1:{self.relationship_type}]->(shared)<-[r2:{self.relationship_type}]-(c2)
        RETURN shared, r1.predicate as c1_relationship, r2.predicate as c2_relationship,
               r1.confidence as c1_confidence, r2.confidence as c2_confidence
        """
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher_query, company1=company1, company2=company2)
            
            shared_connections = []
            for record in result:
                shared_connections.append({
                    "shared_entity": dict(record["shared"]),
                    "company1_relationship": record["c1_relationship"],
                    "company2_relationship": record["c2_relationship"],
                    "company1_confidence": record["c1_confidence"],
                    "company2_confidence": record["c2_confidence"]
                })
            
            return {
                "company1": company1,
                "company2": company2,
                "shared_connections": shared_connections,
                "alignment_score": len(shared_connections)
            }
    
    def get_entity_by_type(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all entities of a specific type"""
        if not self.driver:
            raise RuntimeError("Database connection not established")
        
        cypher_query = f"""
        MATCH (e:{self.entity_label} {{type: $entity_type}})
        RETURN e.name as name, e.type as type, e.created_at as created_at,
               size((e)--()) as connection_count
        ORDER BY connection_count DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher_query, entity_type=entity_type, limit=limit)
            
            return [dict(record) for record in result]
    
    def clear_database(self, confirm: bool = False) -> bool:
        """Clear all data from the database (use with caution!)"""
        if not confirm:
            logger.warning("Database clear not confirmed. Use confirm=True to proceed.")
            return False
        
        if not self.driver:
            logger.error("No database connection established")
            return False
        
        try:
            with self.driver.session(database=self.config.database) as session:
                # Delete all relationships first
                session.run(f"MATCH ()-[r:{self.relationship_type}]-() DELETE r")
                
                # Then delete all entities
                session.run(f"MATCH (e:{self.entity_label}) DELETE e")
                
                logger.info("Database cleared successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def export_to_json(self, output_path: str) -> bool:
        """Export entire graph to JSON format"""
        if not self.driver:
            logger.error("No database connection established")
            return False
        
        try:
            with self.driver.session(database=self.config.database) as session:
                # Export entities
                entity_query = f"MATCH (e:{self.entity_label}) RETURN e"
                entity_result = session.run(entity_query)
                entities = [dict(record["e"]) for record in entity_result]
                
                # Export relationships
                rel_query = f"MATCH (s)-[r:{self.relationship_type}]->(o) RETURN s.name as subject, type(r) as rel_type, properties(r) as rel_props, o.name as object"
                rel_result = session.run(rel_query)
                relationships = []
                
                for record in rel_result:
                    rel_data = dict(record["rel_props"])
                    rel_data.update({
                        "subject": record["subject"],
                        "object": record["object"],
                        "relationship_type": record["rel_type"]
                    })
                    relationships.append(rel_data)
                
                # Export to JSON
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "entities": entities,
                    "relationships": relationships,
                    "stats": {
                        "entity_count": len(entities),
                        "relationship_count": len(relationships)
                    }
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"Graph exported to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return False

def main():
    """Demo the Neo4j ingestion system"""
    
    # Configuration
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j", 
        password="password"  # Change this to your Neo4j password
    )
    
    # Initialize ingestion system
    ingestion = Neo4jIngestion(config)
    
    # Connect to database
    if not ingestion.connect():
        logger.error("Failed to connect to Neo4j. Please check your configuration.")
        return
    
    try:
        # Setup database
        if not ingestion.setup_database():
            logger.error("Failed to setup database")
            return
        
        # Example: Ingest extracted knowledge
        if Path("extracted_knowledge.json").exists():
            logger.info("Ingesting extracted knowledge...")
            stats = ingestion.ingest_json_file("extracted_knowledge.json")
            
            print("\n=== INGESTION STATISTICS ===")
            print(f"Total triplets: {stats.total_triplets}")
            print(f"Successful: {stats.successful_ingestions}")
            print(f"Failed: {stats.failed_ingestions}")
            print(f"Processing time: {stats.processing_time:.2f}s")
            print(f"Unique entities: {stats.unique_entities}")
            print(f"Unique relationships: {stats.unique_relationships}")
            
            if stats.errors:
                print(f"\nErrors ({len(stats.errors)}):")
                for error in stats.errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
        else:
            logger.info("No extracted_knowledge.json found. Run csv_knowledge_extractor.py first.")
            
        # Example queries
        print("\n=== EXAMPLE QUERIES ===")
        
        # Get companies
        companies = ingestion.get_entity_by_type("Company", limit=5)
        print(f"\nTop 5 companies by connections:")
        for company in companies:
            print(f"  {company['name']}: {company['connection_count']} connections")
        
        # Get technologies
        technologies = ingestion.get_entity_by_type("Technology", limit=5)
        print(f"\nTop 5 technologies by connections:")
        for tech in technologies:
            print(f"  {tech['name']}: {tech['connection_count']} connections")
            
    finally:
        # Always disconnect
        ingestion.disconnect()

if __name__ == "__main__":
    main() 