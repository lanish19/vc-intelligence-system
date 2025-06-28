#!/usr/bin/env python3
"""
Advanced Graph Analytics for Venture Capital Intelligence

This module implements Part II of the strategic framework: "Advanced Analytics - Extracting 
Latent Insight and White Space". It provides sophisticated graph analysis capabilities for 
investment intelligence including:

- Section 4: Community Detection for thematic clustering
- Section 5: Subgraph Comparison for thesis alignment analysis  
- Section 6: Centrality Analysis for identifying keystone technologies
- Section 7: Link Prediction for white space discovery

Author: AI Mapping Knowledge Graph System
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import networkx as nx
from neo4j import GraphDatabase
import community as community_louvain
from scipy.spatial.distance import cosine
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CommunityInfo:
    """Information about a detected community"""
    community_id: int
    size: int
    companies: List[str]
    technologies: List[str]
    market_frictions: List[str]
    thesis_concepts: List[str]
    theme_description: str
    modularity_contribution: float

@dataclass
class CentralityScores:
    """Centrality scores for a node"""
    node_name: str
    node_type: str
    degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    closeness_centrality: float

@dataclass
class LinkPrediction:
    """Predicted link between two nodes"""
    node1: str
    node1_type: str
    node2: str
    node2_type: str
    prediction_score: float
    prediction_method: str
    common_neighbors: int
    rationale: str

@dataclass
class SubgraphComparison:
    """Comparison between two company subgraphs"""
    company1: str
    company2: str
    shared_nodes: List[str]
    shared_technologies: List[str]
    shared_frictions: List[str]
    alignment_score: float
    friction_score: float
    thesis_similarity: float
    competitive_overlap: bool

class GraphAnalytics:
    """
    Advanced analytics engine for the venture capital knowledge graph.
    
    Implements the analytical framework from Part II:
    - Community detection using Louvain algorithm
    - Centrality analysis for influence mapping
    - Subgraph comparison for competitive analysis
    - Link prediction for opportunity discovery
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password"):
        """Initialize with Neo4j connection"""
        self.neo4j_uri = neo4j_uri
        self.username = username
        self.password = password
        self.driver = None
        self.graph = None
        
    def connect_neo4j(self) -> bool:
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.username, self.password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
                
            logger.info("Connected to Neo4j for analytics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close_neo4j(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def load_graph_from_neo4j(self) -> bool:
        """
        Load the knowledge graph from Neo4j into NetworkX for analysis.
        This enables the application of standard graph algorithms.
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False
        
        try:
            self.graph = nx.Graph()
            
            with self.driver.session() as session:
                # Load nodes
                node_query = """
                MATCH (n:Entity)
                RETURN n.name as name, n.type as type, labels(n) as labels
                """
                
                result = session.run(node_query)
                for record in result:
                    node_name = record["name"]
                    node_type = record["type"]
                    labels = record["labels"]
                    
                    # Add node with attributes
                    self.graph.add_node(node_name, 
                                      type=node_type, 
                                      labels=labels)
                
                # Load relationships
                rel_query = """
                MATCH (n1:Entity)-[r:RELATES_TO]->(n2:Entity)
                RETURN n1.name as source, n2.name as target, 
                       r.type as rel_type, r.confidence as confidence,
                       r.source_document as source_doc,
                       coalesce(r.inferred, false) as inferred
                """
                
                result = session.run(rel_query)
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    rel_type = record["rel_type"]
                    confidence = record["confidence"] or 1.0
                    source_doc = record["source_doc"]
                    inferred = record["inferred"]
                    
                    # Add edge with attributes
                    self.graph.add_edge(source, target,
                                      rel_type=rel_type,
                                      confidence=confidence,
                                      source_document=source_doc,
                                      inferred=inferred)
                
                logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
                return True
                
        except Exception as e:
            logger.error(f"Error loading graph from Neo4j: {e}")
            return False
    
    def detect_communities_louvain(self) -> List[CommunityInfo]:
        """
        Section 4: Community Detection using the Louvain algorithm.
        
        Identifies thematic clusters of companies, technologies, and concepts
        that are densely interconnected, revealing emergent investment themes.
        
        Returns:
            List of CommunityInfo objects describing each detected community
        """
        if not self.graph:
            logger.error("Graph not loaded. Call load_graph_from_neo4j() first.")
            return []
        
        logger.info("Running Louvain community detection...")
        
        try:
            # Run Louvain algorithm
            communities = community_louvain.best_partition(self.graph)
            modularity = community_louvain.modularity(communities, self.graph)
            
            logger.info(f"Detected {len(set(communities.values()))} communities with modularity {modularity:.3f}")
            
            # Analyze each community
            community_infos = []
            community_nodes = defaultdict(list)
            
            # Group nodes by community
            for node, comm_id in communities.items():
                community_nodes[comm_id].append(node)
            
            for comm_id, nodes in community_nodes.items():
                if len(nodes) < 3:  # Skip very small communities
                    continue
                
                # Categorize nodes by type
                companies = []
                technologies = []
                market_frictions = []
                thesis_concepts = []
                
                for node in nodes:
                    node_data = self.graph.nodes[node]
                    node_type = node_data.get('type', 'Unknown')
                    
                    if node_type == 'Company':
                        companies.append(node)
                    elif node_type == 'Technology':
                        technologies.append(node)
                    elif node_type == 'MarketFriction':
                        market_frictions.append(node)
                    elif node_type == 'ThesisConcept':
                        thesis_concepts.append(node)
                
                # Generate theme description
                theme = self._generate_theme_description(
                    companies, technologies, market_frictions, thesis_concepts
                )
                
                # Calculate community's contribution to overall modularity
                subgraph = self.graph.subgraph(nodes)
                internal_edges = subgraph.number_of_edges()
                total_degree = sum(dict(self.graph.degree(nodes)).values())
                expected_internal = (total_degree ** 2) / (4 * self.graph.number_of_edges())
                modularity_contrib = (internal_edges - expected_internal) / self.graph.number_of_edges()
                
                community_info = CommunityInfo(
                    community_id=comm_id,
                    size=len(nodes),
                    companies=companies,
                    technologies=technologies,
                    market_frictions=market_frictions,
                    thesis_concepts=thesis_concepts,
                    theme_description=theme,
                    modularity_contribution=modularity_contrib
                )
                
                community_infos.append(community_info)
            
            # Sort by size (largest first)
            community_infos.sort(key=lambda x: x.size, reverse=True)
            
            logger.info(f"Analyzed {len(community_infos)} significant communities")
            return community_infos
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return []
    
    def _generate_theme_description(self, companies: List[str], technologies: List[str], 
                                  market_frictions: List[str], thesis_concepts: List[str]) -> str:
        """Generate a descriptive theme for a community"""
        
        # Analyze technology patterns
        tech_keywords = []
        for tech in technologies:
            if 'AI' in tech or 'ML' in tech:
                tech_keywords.append('AI/ML')
            if 'Autonomous' in tech or 'Automation' in tech:
                tech_keywords.append('Autonomy')
            if 'Cyber' in tech or 'Security' in tech:
                tech_keywords.append('Cybersecurity')
            if 'Edge' in tech:
                tech_keywords.append('Edge Computing')
            if 'Cloud' in tech:
                tech_keywords.append('Cloud')
        
        # Analyze friction patterns
        friction_keywords = []
        for friction in market_frictions:
            if 'threat' in friction.lower() or 'security' in friction.lower():
                friction_keywords.append('Security Threats')
            if 'manual' in friction.lower() or 'automation' in friction.lower():
                friction_keywords.append('Manual Processes')
            if 'scale' in friction.lower() or 'scalability' in friction.lower():
                friction_keywords.append('Scalability')
        
        # Generate theme
        tech_counter = Counter(tech_keywords)
        friction_counter = Counter(friction_keywords)
        
        primary_tech = tech_counter.most_common(1)[0][0] if tech_counter else "Technology"
        primary_friction = friction_counter.most_common(1)[0][0] if friction_counter else "Market Gap"
        
        if len(companies) >= 5:
            theme = f"Large {primary_tech} cluster addressing {primary_friction}"
        elif len(companies) >= 3:
            theme = f"{primary_tech} companies targeting {primary_friction}"
        else:
            theme = f"Emerging {primary_tech} theme"
        
        return theme
    
    def calculate_centrality_measures(self) -> List[CentralityScores]:
        """
        Section 6: Centrality Analysis for identifying keystone technologies and influential nodes.
        
        Calculates multiple centrality measures to identify:
        - Keystone technologies (high betweenness centrality)
        - Popular concepts (high degree centrality)  
        - Influential players (high eigenvector centrality)
        
        Returns:
            List of CentralityScores for all nodes
        """
        if not self.graph:
            logger.error("Graph not loaded. Call load_graph_from_neo4j() first.")
            return []
        
        logger.info("Calculating centrality measures...")
        
        try:
            # Calculate all centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
            
            # Handle potential issues with eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                logger.warning("Eigenvector centrality failed, using degree centrality as fallback")
                eigenvector_centrality = degree_centrality
            
            pagerank = nx.pagerank(self.graph)
            
            # Closeness centrality (sample for large graphs)
            if self.graph.number_of_nodes() > 1000:
                # Sample nodes for closeness centrality to avoid performance issues
                sample_nodes = list(self.graph.nodes())[:500]
                closeness_sample = nx.closeness_centrality(self.graph.subgraph(sample_nodes))
                closeness_centrality = {node: closeness_sample.get(node, 0) for node in self.graph.nodes()}
            else:
                closeness_centrality = nx.closeness_centrality(self.graph)
            
            # Combine results
            centrality_scores = []
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_type = node_data.get('type', 'Unknown')
                
                scores = CentralityScores(
                    node_name=node,
                    node_type=node_type,
                    degree_centrality=degree_centrality.get(node, 0),
                    betweenness_centrality=betweenness_centrality.get(node, 0),
                    eigenvector_centrality=eigenvector_centrality.get(node, 0),
                    pagerank=pagerank.get(node, 0),
                    closeness_centrality=closeness_centrality.get(node, 0)
                )
                
                centrality_scores.append(scores)
            
            logger.info(f"Calculated centrality measures for {len(centrality_scores)} nodes")
            return centrality_scores
            
        except Exception as e:
            logger.error(f"Error calculating centrality measures: {e}")
            return []
    
    def analyze_keystone_technologies(self, centrality_scores: List[CentralityScores]) -> List[Dict[str, Any]]:
        """
        Analyze keystone technologies based on centrality measures.
        
        Keystone technologies are identified by high betweenness centrality,
        indicating they act as bridges between different parts of the ecosystem.
        """
        # Filter for technology nodes
        tech_scores = [s for s in centrality_scores if s.node_type == 'Technology']
        
        # Sort by betweenness centrality
        tech_scores.sort(key=lambda x: x.betweenness_centrality, reverse=True)
        
        keystone_techs = []
        for score in tech_scores[:10]:  # Top 10 keystone technologies
            # Find connected companies
            connected_companies = []
            if self.graph and score.node_name in self.graph:
                neighbors = list(self.graph.neighbors(score.node_name))
                connected_companies = [n for n in neighbors if 
                                     self.graph.nodes[n].get('type') == 'Company']
            
            keystone_techs.append({
                'technology': score.node_name,
                'betweenness_centrality': score.betweenness_centrality,
                'degree_centrality': score.degree_centrality,
                'pagerank': score.pagerank,
                'connected_companies': len(connected_companies),
                'companies': connected_companies[:5],  # Sample companies
                'keystone_score': (score.betweenness_centrality * 0.6 + 
                                 score.degree_centrality * 0.4)
            })
        
        return keystone_techs
    
    def compare_company_subgraphs(self, company1: str, company2: str) -> Optional[SubgraphComparison]:
        """
        Section 5: Subgraph Comparison for analyzing thesis alignment and competitive overlap.
        
        Compares the 2-hop subgraphs of two companies to identify:
        - Shared technologies and approaches
        - Overlapping market frictions
        - Thesis alignment vs friction
        """
        if not self.graph:
            logger.error("Graph not loaded. Call load_graph_from_neo4j() first.")
            return None
        
        if company1 not in self.graph or company2 not in self.graph:
            logger.warning(f"One or both companies not found in graph: {company1}, {company2}")
            return None
        
        try:
            # Get 2-hop subgraphs for each company
            subgraph1_nodes = set([company1])
            subgraph2_nodes = set([company2])
            
            # Add 1-hop neighbors
            for neighbor in self.graph.neighbors(company1):
                subgraph1_nodes.add(neighbor)
                # Add 2-hop neighbors
                for second_neighbor in self.graph.neighbors(neighbor):
                    subgraph1_nodes.add(second_neighbor)
            
            for neighbor in self.graph.neighbors(company2):
                subgraph2_nodes.add(neighbor)
                # Add 2-hop neighbors  
                for second_neighbor in self.graph.neighbors(neighbor):
                    subgraph2_nodes.add(second_neighbor)
            
            # Find shared nodes
            shared_nodes = list(subgraph1_nodes.intersection(subgraph2_nodes))
            shared_nodes.remove(company1) if company1 in shared_nodes else None
            shared_nodes.remove(company2) if company2 in shared_nodes else None
            
            # Categorize shared nodes
            shared_technologies = []
            shared_frictions = []
            shared_thesis = []
            
            for node in shared_nodes:
                node_type = self.graph.nodes[node].get('type', 'Unknown')
                if node_type == 'Technology':
                    shared_technologies.append(node)
                elif node_type == 'MarketFriction':
                    shared_frictions.append(node)
                elif node_type == 'ThesisConcept':
                    shared_thesis.append(node)
            
            # Calculate similarity scores
            total_nodes1 = len(subgraph1_nodes)
            total_nodes2 = len(subgraph2_nodes)
            shared_count = len(shared_nodes)
            
            # Jaccard similarity for overall alignment
            union_size = len(subgraph1_nodes.union(subgraph2_nodes))
            alignment_score = shared_count / union_size if union_size > 0 else 0
            
            # Friction score (how much they compete in same problems)
            friction_score = len(shared_frictions) / max(1, (len(shared_frictions) + 
                            len(set([n for n in subgraph1_nodes if self.graph.nodes[n].get('type') == 'MarketFriction'])) +
                            len(set([n for n in subgraph2_nodes if self.graph.nodes[n].get('type') == 'MarketFriction']))))
            
            # Thesis similarity
            thesis_similarity = len(shared_thesis) / max(1, (len(shared_thesis) +
                              len(set([n for n in subgraph1_nodes if self.graph.nodes[n].get('type') == 'ThesisConcept'])) +
                              len(set([n for n in subgraph2_nodes if self.graph.nodes[n].get('type') == 'ThesisConcept']))))
            
            # Determine if they're competitive (high friction overlap)
            competitive_overlap = (len(shared_frictions) >= 2 and friction_score > 0.3)
            
            comparison = SubgraphComparison(
                company1=company1,
                company2=company2,
                shared_nodes=shared_nodes,
                shared_technologies=shared_technologies,
                shared_frictions=shared_frictions,
                alignment_score=alignment_score,
                friction_score=friction_score,
                thesis_similarity=thesis_similarity,
                competitive_overlap=competitive_overlap
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing subgraphs: {e}")
            return None
    
    def predict_links_adamic_adar(self, node_type_filter: Optional[str] = None) -> List[LinkPrediction]:
        """
        Section 7: Link Prediction using Adamic-Adar index for white space discovery.
        
        Identifies potential connections between nodes based on shared neighbors,
        weighted by the rarity of those neighbors (Adamic-Adar heuristic).
        
        Args:
            node_type_filter: Filter predictions to specific node types (e.g., 'Company')
            
        Returns:
            List of LinkPrediction objects ranked by score
        """
        if not self.graph:
            logger.error("Graph not loaded. Call load_graph_from_neo4j() first.")
            return []
        
        logger.info("Predicting links using Adamic-Adar index...")
        
        try:
            predictions = []
            
            # Get nodes for prediction
            nodes = list(self.graph.nodes())
            if node_type_filter:
                nodes = [n for n in nodes if self.graph.nodes[n].get('type') == node_type_filter]
            
            # Calculate Adamic-Adar scores for non-connected node pairs
            for node1, node2 in combinations(nodes, 2):
                if not self.graph.has_edge(node1, node2):  # Only predict missing links
                    
                    # Get node types
                    node1_type = self.graph.nodes[node1].get('type', 'Unknown')
                    node2_type = self.graph.nodes[node2].get('type', 'Unknown')
                    
                    # Find common neighbors
                    neighbors1 = set(self.graph.neighbors(node1))
                    neighbors2 = set(self.graph.neighbors(node2))
                    common_neighbors = neighbors1.intersection(neighbors2)
                    
                    if len(common_neighbors) > 0:  # Only consider pairs with common neighbors
                        # Calculate Adamic-Adar score
                        aa_score = 0
                        for common_neighbor in common_neighbors:
                            degree = self.graph.degree(common_neighbor)
                            if degree > 1:  # Avoid division by zero
                                aa_score += 1 / np.log(degree)
                        
                        # Generate rationale based on node types and common neighbors
                        rationale = self._generate_link_rationale(
                            node1, node1_type, node2, node2_type, common_neighbors
                        )
                        
                        prediction = LinkPrediction(
                            node1=node1,
                            node1_type=node1_type,
                            node2=node2,
                            node2_type=node2_type,
                            prediction_score=aa_score,
                            prediction_method="Adamic-Adar",
                            common_neighbors=len(common_neighbors),
                            rationale=rationale
                        )
                        
                        predictions.append(prediction)
            
            # Sort by prediction score (highest first)
            predictions.sort(key=lambda x: x.prediction_score, reverse=True)
            
            # Filter for strategic relevance (remove very low scores)
            significant_predictions = [p for p in predictions if p.prediction_score > 0.1][:50]
            
            logger.info(f"Generated {len(significant_predictions)} significant link predictions")
            return significant_predictions
            
        except Exception as e:
            logger.error(f"Error in link prediction: {e}")
            return []
    
    def _generate_link_rationale(self, node1: str, node1_type: str, node2: str, node2_type: str, 
                               common_neighbors: Set[str]) -> str:
        """Generate business rationale for predicted link"""
        
        # Analyze common neighbor types
        neighbor_types = [self.graph.nodes[n].get('type', 'Unknown') for n in common_neighbors]
        type_counts = Counter(neighbor_types)
        
        if node1_type == 'Company' and node2_type == 'Company':
            if 'Technology' in type_counts:
                return f"Both companies use similar technologies ({type_counts['Technology']} shared), suggesting potential partnership or competitive overlap."
            elif 'MarketFriction' in type_counts:
                return f"Both companies address similar market problems ({type_counts['MarketFriction']} shared), indicating competitive landscape."
            else:
                return "Companies share strategic context, suggesting market adjacency."
                
        elif node1_type == 'Company' and node2_type == 'Technology':
            return f"Company {node1} may benefit from adopting {node2} technology based on shared strategic context."
            
        elif node1_type == 'Company' and node2_type == 'MarketFriction':
            return f"Company {node1} may expand to address {node2} based on existing capabilities."
            
        elif node1_type == 'Technology' and node2_type == 'Technology':
            return f"Technologies {node1} and {node2} are frequently used together, suggesting complementary capabilities."
            
        else:
            return f"Strategic connection suggested by {len(common_neighbors)} shared contexts."
    
    def generate_investment_insights(self, communities: List[CommunityInfo], 
                                   centrality_scores: List[CentralityScores],
                                   link_predictions: List[LinkPrediction]) -> Dict[str, Any]:
        """
        Generate comprehensive investment insights by combining all analytical results.
        
        This synthesizes the framework's analytics into actionable intelligence.
        """
        insights = {
            'executive_summary': {},
            'thematic_clusters': [],
            'keystone_technologies': [],
            'white_space_opportunities': [],
            'competitive_intelligence': [],
            'investment_recommendations': []
        }
        
        # Executive Summary
        insights['executive_summary'] = {
            'total_communities': len(communities),
            'largest_community_size': max([c.size for c in communities]) if communities else 0,
            'most_central_technology': max(centrality_scores, key=lambda x: x.betweenness_centrality if x.node_type == 'Technology' else 0).node_name if centrality_scores else None,
            'top_white_space_score': max([p.prediction_score for p in link_predictions]) if link_predictions else 0,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Thematic Clusters (Top 5)
        for i, community in enumerate(communities[:5]):
            insights['thematic_clusters'].append({
                'rank': i + 1,
                'theme': community.theme_description,
                'size': community.size,
                'company_count': len(community.companies),
                'key_companies': community.companies[:3],
                'core_technologies': community.technologies[:3],
                'market_opportunity': community.market_frictions[:2],
                'investment_thesis': community.thesis_concepts[:2] if community.thesis_concepts else ["Emerging theme"]
            })
        
        # Keystone Technologies
        keystone_techs = self.analyze_keystone_technologies(centrality_scores)
        insights['keystone_technologies'] = keystone_techs[:10]
        
        # White Space Opportunities (Top 10 predictions)
        for i, prediction in enumerate(link_predictions[:10]):
            insights['white_space_opportunities'].append({
                'rank': i + 1,
                'opportunity_type': f"{prediction.node1_type} - {prediction.node2_type}",
                'description': f"{prediction.node1} → {prediction.node2}",
                'score': prediction.prediction_score,
                'rationale': prediction.rationale,
                'common_context': prediction.common_neighbors
            })
        
        # Investment Recommendations
        recommendations = []
        
        # Technology convergence opportunities
        if communities:
            largest_community = communities[0]
            if len(largest_community.technologies) >= 2:
                recommendations.append({
                    'type': 'Technology Convergence',
                    'priority': 'High',
                    'description': f"Invest in companies combining {', '.join(largest_community.technologies[:2])}",
                    'market_size': f"{len(largest_community.companies)} companies active",
                    'rationale': f"Largest thematic cluster with strong growth potential"
                })
        
        # Keystone technology positions
        if keystone_techs:
            top_keystone = keystone_techs[0]
            recommendations.append({
                'type': 'Keystone Technology',
                'priority': 'High',
                'description': f"Secure position in {top_keystone['technology']} ecosystem",
                'market_size': f"{top_keystone['connected_companies']} connected companies",
                'rationale': "High centrality indicates critical enabling technology"
            })
        
        # White space exploitation
        company_predictions = [p for p in link_predictions if p.node1_type == 'Company' and p.node2_type == 'MarketFriction']
        if company_predictions:
            top_expansion = company_predictions[0]
            recommendations.append({
                'type': 'Market Expansion',
                'priority': 'Medium',
                'description': f"Support {top_expansion.node1} expansion into {top_expansion.node2}",
                'market_size': "New market opportunity",
                'rationale': top_expansion.rationale
            })
        
        insights['investment_recommendations'] = recommendations
        
        return insights
    
    def save_results(self, results: Dict[str, Any], filename: str = "graph_analytics_results.json"):
        """Save analysis results to JSON file"""
        try:
            # Convert dataclass objects to dictionaries
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                    serializable_results[key] = [asdict(item) for item in value]
                else:
                    serializable_results[key] = value
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main execution function for graph analytics"""
    analytics = GraphAnalytics()
    
    try:
        # Connect to Neo4j and load graph
        if not analytics.connect_neo4j():
            logger.error("Failed to connect to Neo4j")
            return
        
        if not analytics.load_graph_from_neo4j():
            logger.error("Failed to load graph from Neo4j")
            return
        
        logger.info("=== RUNNING ADVANCED GRAPH ANALYTICS ===")
        
        # Section 4: Community Detection
        logger.info("\n1. COMMUNITY DETECTION")
        communities = analytics.detect_communities_louvain()
        
        if communities:
            logger.info(f"Found {len(communities)} communities:")
            for i, community in enumerate(communities[:5]):
                logger.info(f"   {i+1}. {community.theme_description} ({community.size} nodes)")
                logger.info(f"      Companies: {len(community.companies)}, Technologies: {len(community.technologies)}")
        
        # Section 6: Centrality Analysis  
        logger.info("\n2. CENTRALITY ANALYSIS")
        centrality_scores = analytics.calculate_centrality_measures()
        
        if centrality_scores:
            # Show top nodes by different centrality measures
            by_degree = sorted(centrality_scores, key=lambda x: x.degree_centrality, reverse=True)[:5]
            by_betweenness = sorted(centrality_scores, key=lambda x: x.betweenness_centrality, reverse=True)[:5]
            
            logger.info("Top nodes by degree centrality:")
            for score in by_degree:
                logger.info(f"   {score.node_name} ({score.node_type}): {score.degree_centrality:.3f}")
            
            logger.info("Top nodes by betweenness centrality (keystone candidates):")
            for score in by_betweenness:
                logger.info(f"   {score.node_name} ({score.node_type}): {score.betweenness_centrality:.3f}")
        
        # Section 7: Link Prediction
        logger.info("\n3. LINK PREDICTION")
        link_predictions = analytics.predict_links_adamic_adar()
        
        if link_predictions:
            logger.info(f"Generated {len(link_predictions)} link predictions:")
            for i, pred in enumerate(link_predictions[:5]):
                logger.info(f"   {i+1}. {pred.node1} → {pred.node2} (score: {pred.prediction_score:.3f})")
                logger.info(f"      {pred.rationale}")
        
        # Generate comprehensive insights
        logger.info("\n4. GENERATING INVESTMENT INSIGHTS")
        insights = analytics.generate_investment_insights(communities, centrality_scores, link_predictions)
        
        # Display key insights
        logger.info("\n=== KEY INSIGHTS ===")
        logger.info(f"Identified {insights['executive_summary']['total_communities']} thematic clusters")
        logger.info(f"Largest cluster: {insights['executive_summary']['largest_community_size']} entities")
        logger.info(f"Most central technology: {insights['executive_summary']['most_central_technology']}")
        
        logger.info("\nTop Investment Recommendations:")
        for i, rec in enumerate(insights['investment_recommendations']):
            logger.info(f"   {i+1}. {rec['type']}: {rec['description']} (Priority: {rec['priority']})")
        
        # Save all results
        all_results = {
            'communities': communities,
            'centrality_scores': centrality_scores,
            'link_predictions': link_predictions,
            'investment_insights': insights
        }
        
        analytics.save_results(all_results)
        
        logger.info("\n=== ANALYSIS COMPLETE ===")
        logger.info("Results saved to graph_analytics_results.json")
        
    finally:
        analytics.close_neo4j()

if __name__ == "__main__":
    main() 