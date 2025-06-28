#!/usr/bin/env python3
"""
Main Knowledge Graph Pipeline

This script implements the complete end-to-end pipeline from the strategic framework,
orchestrating all components to transform raw AI mapping data into actionable VC intelligence.

Implements:
- Data extraction and knowledge graph creation
- Advanced analytics (community detection, centrality, link prediction)  
- Evaluation and quality control
- Visualization and reporting

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

# Import our framework components
from csv_knowledge_extractor import VCOntologyExtractor
from neo4j_ingestion import Neo4jIngestion
from graph_analytics import GraphAnalytics
from evaluation_system import EvaluationSystem
from graph_visualizer import GraphVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VCIntelligencePipeline:
    """
    Complete VC intelligence pipeline implementing the strategic framework.
    
    This orchestrates the transformation from raw data to actionable insights:
    1. Knowledge extraction (Section 1-2)
    2. Meta-graph creation (Section 3)
    3. Advanced analytics (Section 4-7)  
    4. Quality evaluation (Section 8)
    5. Visualization (Section 9)
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        """Initialize the pipeline components"""
        
        # Core components
        self.extractor = VCOntologyExtractor()
        self.neo4j = Neo4jIngestion(neo4j_uri, neo4j_user, neo4j_password)
        self.analytics = GraphAnalytics(neo4j_uri, neo4j_user, neo4j_password)
        self.evaluator = EvaluationSystem()
        self.visualizer = GraphVisualizer()
        
        # Pipeline state
        self.results = {}
        self.execution_time = {}
        
    def run_complete_pipeline(self, csv_path: str = "raw ai mapping data.csv",
                            output_dir: str = "output") -> Dict[str, Any]:
        """
        Execute the complete pipeline from CSV to intelligence insights.
        
        Args:
            csv_path: Path to the AI mapping CSV data
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing all results and analytics
        """
        
        logger.info("🚀 Starting VC Intelligence Pipeline")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Phase 1: Knowledge Extraction
            logger.info("📊 Phase 1: Knowledge Extraction")
            extraction_start = time.time()
            
            triplets = self.extractor.process_csv(csv_path)
            triplet_file = output_path / "extracted_triplets.json"
            self.extractor.save_triplets(triplets, str(triplet_file))
            
            self.execution_time['extraction'] = time.time() - extraction_start
            self.results['triplets_count'] = len(triplets)
            
            logger.info(f"✅ Extracted {len(triplets)} knowledge triplets")
            
            # Phase 2: Meta-Graph Creation  
            logger.info("🔗 Phase 2: Meta-Graph Creation")
            ingestion_start = time.time()
            
            # Connect to Neo4j and ingest data
            if not self.neo4j.connect():
                raise RuntimeError("Failed to connect to Neo4j")
            
            # Create indexes for performance
            self.neo4j.create_indexes()
            
            # Ingest triplets in batches
            success = self.neo4j.ingest_triplets_from_json(str(triplet_file))
            if not success:
                raise RuntimeError("Failed to ingest triplets into Neo4j")
            
            # Get database statistics
            db_stats = self.neo4j.get_database_stats()
            self.results['database_stats'] = db_stats
            
            self.execution_time['ingestion'] = time.time() - ingestion_start
            
            logger.info(f"✅ Created meta-graph: {db_stats['total_nodes']} nodes, {db_stats['total_relationships']} relationships")
            
            # Phase 3: Advanced Analytics
            logger.info("🧠 Phase 3: Advanced Analytics")
            analytics_start = time.time()
            
            # Connect analytics module to Neo4j
            if not self.analytics.connect_neo4j():
                raise RuntimeError("Failed to connect analytics to Neo4j")
            
            # Load graph for analysis
            if not self.analytics.load_graph_from_neo4j():
                raise RuntimeError("Failed to load graph for analytics")
            
            # Run community detection (Section 4)
            logger.info("🔍 Running community detection...")
            communities = self.analytics.detect_communities_louvain()
            self.results['communities'] = [
                {
                    'id': c.community_id,
                    'size': c.size,
                    'theme': c.theme_description,
                    'companies': c.companies[:5],  # Top 5 companies
                    'technologies': c.technologies[:5]  # Top 5 technologies
                }
                for c in communities[:10]  # Top 10 communities
            ]
            
            # Run centrality analysis (Section 6)
            logger.info("📈 Calculating centrality measures...")
            centrality_scores = self.analytics.calculate_centrality_measures()
            
            # Identify keystone technologies
            keystone_tech = self.analytics.analyze_keystone_technologies(centrality_scores)
            self.results['keystone_technologies'] = keystone_tech[:10]  # Top 10
            
            # Top influential nodes by centrality
            top_nodes = sorted(centrality_scores, key=lambda x: x.pagerank, reverse=True)[:15]
            self.results['most_influential_nodes'] = [
                {
                    'name': node.node_name,
                    'type': node.node_type,
                    'pagerank': node.pagerank,
                    'betweenness': node.betweenness_centrality,
                    'degree': node.degree_centrality
                }
                for node in top_nodes
            ]
            
            # Run link prediction for white space analysis (Section 7)
            logger.info("🎯 Discovering white space opportunities...")
            link_predictions = self.analytics.predict_links_adamic_adar()
            
            # Filter for high-value predictions
            high_value_predictions = [p for p in link_predictions if p.prediction_score > 0.1][:20]
            self.results['white_space_opportunities'] = [
                {
                    'opportunity': f"{pred.node1} -> {pred.node2}",
                    'node1': pred.node1,
                    'node1_type': pred.node1_type,
                    'node2': pred.node2, 
                    'node2_type': pred.node2_type,
                    'score': pred.prediction_score,
                    'rationale': pred.rationale
                }
                for pred in high_value_predictions
            ]
            
            # Generate investment insights
            investment_insights = self.analytics.generate_investment_insights(
                communities, centrality_scores, link_predictions
            )
            self.results['investment_insights'] = investment_insights
            
            self.execution_time['analytics'] = time.time() - analytics_start
            
            logger.info(f"✅ Analytics complete: {len(communities)} communities, {len(keystone_tech)} keystone technologies")
            
            # Phase 4: Visualization
            logger.info("📊 Phase 4: Creating Visualizations")
            viz_start = time.time()
            
            # Load graph for visualization
            self.visualizer.load_networkx_graph(self.analytics.graph)
            
            # Create interactive visualization
            interactive_fig = self.visualizer.create_interactive_plot(
                title="AI Defense Technology Knowledge Graph"
            )
            
            # Save visualizations
            interactive_file = output_path / "knowledge_graph_interactive.html"
            self.visualizer.save_interactive_html(interactive_fig, str(interactive_file))
            
            # Create static visualization
            static_fig = self.visualizer.create_matplotlib_plot(
                title="AI Defense Technology Knowledge Graph"
            )
            static_file = output_path / "knowledge_graph_static.png"
            self.visualizer.save_static_image(static_fig, str(static_file))
            
            self.execution_time['visualization'] = time.time() - viz_start
            
            logger.info("✅ Visualizations created")
            
            # Phase 5: Generate Reports
            logger.info("📋 Phase 5: Generating Reports")
            
            # Save detailed results
            detailed_results = {
                'pipeline_metadata': {
                    'execution_time': self.execution_time,
                    'total_time': time.time() - start_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'data_source': csv_path
                },
                'knowledge_extraction': {
                    'total_triplets': self.results['triplets_count'],
                    'triplet_file': str(triplet_file)
                },
                'meta_graph': self.results['database_stats'],
                'analytics': {
                    'communities': self.results['communities'],
                    'keystone_technologies': self.results['keystone_technologies'],
                    'influential_nodes': self.results['most_influential_nodes'],
                    'white_space_opportunities': self.results['white_space_opportunities']
                },
                'investment_insights': self.results['investment_insights'],
                'visualizations': {
                    'interactive': str(interactive_file),
                    'static': str(static_file)
                }
            }
            
            # Save comprehensive results
            results_file = output_path / "vc_intelligence_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            # Generate executive summary
            self._generate_executive_summary(detailed_results, output_path)
            
            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"🎉 Pipeline completed successfully in {total_time:.2f} seconds")
            logger.info(f"📁 Results saved to: {output_path}")
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
        
        finally:
            # Cleanup connections
            self.neo4j.close()
            self.analytics.close_neo4j()
    
    def _generate_executive_summary(self, results: Dict[str, Any], output_path: Path):
        """Generate an executive summary of the analysis"""
        
        summary = f"""
# VC Intelligence Pipeline - Executive Summary

**Generated**: {results['pipeline_metadata']['timestamp']}
**Execution Time**: {results['pipeline_metadata']['total_time']:.2f} seconds
**Data Source**: {results['pipeline_metadata']['data_source']}

## Knowledge Graph Overview

- **Total Entities**: {results['meta_graph']['total_nodes']:,}
- **Total Relationships**: {results['meta_graph']['total_relationships']:,}
- **Knowledge Triplets**: {results['knowledge_extraction']['total_triplets']:,}

### Entity Distribution
"""
        
        if 'node_counts' in results['meta_graph']:
            for entity_type, count in results['meta_graph']['node_counts'].items():
                summary += f"- **{entity_type}**: {count:,}\n"
        
        summary += f"""

## Investment Intelligence Insights

### Thematic Clusters
Detected **{len(results['analytics']['communities'])}** major investment themes:

"""
        
        for i, community in enumerate(results['analytics']['communities'][:5], 1):
            summary += f"**{i}. {community['theme']}** ({community['size']} entities)\n"
            if community['companies']:
                summary += f"   - Key Companies: {', '.join(community['companies'])}\n"
            if community['technologies']:
                summary += f"   - Core Technologies: {', '.join(community['technologies'])}\n"
            summary += "\n"
        
        summary += """
### Keystone Technologies
Most influential technologies in the ecosystem:

"""
        
        for i, tech in enumerate(results['analytics']['keystone_technologies'][:5], 1):
            summary += f"**{i}. {tech['technology']}**\n"
            summary += f"   - Influence Score: {tech['influence_score']:.3f}\n"
            summary += f"   - Connected Companies: {tech['connected_companies']}\n"
            summary += f"   - Strategic Importance: {tech['strategic_importance']}\n\n"
        
        summary += """
### White Space Opportunities
High-potential gaps and opportunities identified:

"""
        
        for i, opp in enumerate(results['analytics']['white_space_opportunities'][:5], 1):
            summary += f"**{i}. {opp['opportunity']}**\n"
            summary += f"   - Opportunity Score: {opp['score']:.3f}\n"
            summary += f"   - Rationale: {opp['rationale']}\n\n"
        
        summary += f"""
## Strategic Recommendations

{results['investment_insights'].get('strategic_recommendations', 'Advanced recommendations available in detailed analysis.')}

---

*This analysis was generated using the VC Intelligence Knowledge Graph System*
*For detailed analytics, see: vc_intelligence_results.json*
*Interactive visualization: knowledge_graph_interactive.html*
"""
        
        # Save executive summary
        summary_file = output_path / "executive_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"📊 Executive summary saved to: {summary_file}")

def main():
    """Main execution function"""
    
    print("🎯 VC Intelligence Knowledge Graph System")
    print("=" * 50)
    print("Implementing Strategic Framework for Venture Capital Intelligence")
    print("Using Knowledge Graphs for Meta-Analysis and Opportunity Discovery")
    print("=" * 50)
    
    # Initialize and run pipeline
    pipeline = VCIntelligencePipeline()
    
    try:
        results = pipeline.run_complete_pipeline()
        
        print("\n🎉 SUCCESS: Pipeline completed successfully!")
        print(f"📊 Processed {results['knowledge_extraction']['total_triplets']} knowledge triplets")
        print(f"🔗 Created graph with {results['meta_graph']['total_nodes']} nodes")
        print(f"🧠 Identified {len(results['analytics']['communities'])} investment themes")
        print(f"📁 Results saved to: output/")
        
        # Print key insights
        print("\n🔍 KEY INSIGHTS:")
        print("-" * 30)
        
        if results['analytics']['keystone_technologies']:
            top_tech = results['analytics']['keystone_technologies'][0]
            print(f"🔧 Top Keystone Technology: {top_tech['technology']}")
        
        if results['analytics']['white_space_opportunities']:
            top_opp = results['analytics']['white_space_opportunities'][0]
            print(f"🎯 Top White Space Opportunity: {top_opp['opportunity']}")
        
        if results['analytics']['communities']:
            largest_community = max(results['analytics']['communities'], key=lambda x: x['size'])
            print(f"🏢 Largest Investment Theme: {largest_community['theme']} ({largest_community['size']} entities)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check Neo4j is running and accessible")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 