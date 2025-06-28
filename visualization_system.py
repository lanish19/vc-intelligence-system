#!/usr/bin/env python3
"""
Advanced Visualization System for Strategic Exploration

This module implements Section 9 of the strategic framework: "Advanced Visualization 
for Strategic Exploration". It provides sophisticated graph visualization capabilities 
for the venture capital knowledge graph, enabling interactive analysis and insights 
discovery.

Key Features:
- Interactive network visualization
- Community detection visualization
- Centrality analysis display
- Multi-level filtering and exploration
- Export capabilities for reports

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import colorsys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    node_size_multiplier: float = 10
    edge_width_multiplier: float = 2
    color_scheme: str = "Set3"
    layout_algorithm: str = "spring"
    show_labels: bool = True
    interactive: bool = True

class GraphVisualizer:
    """
    Advanced graph visualization system for VC intelligence.
    
    Implements Section 9 framework requirements:
    - Interactive network exploration
    - Community visualization
    - Centrality highlighting
    - Multi-dimensional analysis
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer with configuration"""
        self.config = config or VisualizationConfig()
        self.graph = None
        self.pos = None
        self.color_map = {}
        
    def load_graph_from_neo4j_data(self, nodes: List[Dict], edges: List[Dict]):
        """Load graph from Neo4j export data"""
        self.graph = nx.Graph()
        
        # Add nodes
        for node in nodes:
            self.graph.add_node(
                node['name'],
                type=node.get('type', 'Unknown'),
                **{k: v for k, v in node.items() if k != 'name'}
            )
        
        # Add edges
        for edge in edges:
            self.graph.add_edge(
                edge['source'],
                edge['target'],
                rel_type=edge.get('rel_type', 'relates_to'),
                confidence=edge.get('confidence', 1.0),
                **{k: v for k, v in edge.items() if k not in ['source', 'target']}
            )
        
        logger.info(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Generate layout
        self._generate_layout()
        
    def _generate_layout(self):
        """Generate node positions using specified layout algorithm"""
        if not self.graph:
            return
        
        logger.info(f"Generating {self.config.layout_algorithm} layout...")
        
        if self.config.layout_algorithm == "spring":
            self.pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif self.config.layout_algorithm == "circular":
            self.pos = nx.circular_layout(self.graph)
        elif self.config.layout_algorithm == "kamada_kawai":
            self.pos = nx.kamada_kawai_layout(self.graph)
        else:
            self.pos = nx.spring_layout(self.graph)
    
    def _generate_color_scheme(self, categories: List[str]) -> Dict[str, str]:
        """Generate distinct colors for categories"""
        colors = px.colors.qualitative.Set3
        color_map = {}
        
        for i, category in enumerate(categories):
            color_map[category] = colors[i % len(colors)]
        
        return color_map
    
    def create_interactive_network(self, title: str = "VC Knowledge Graph", 
                                 filter_by_type: Optional[List[str]] = None,
                                 highlight_nodes: Optional[List[str]] = None) -> go.Figure:
        """
        Create an interactive network visualization.
        
        Args:
            title: Chart title
            filter_by_type: Filter nodes by type (optional)
            highlight_nodes: List of nodes to highlight (optional)
            
        Returns:
            Plotly figure object
        """
        if not self.graph or not self.pos:
            raise ValueError("Graph not loaded. Call load_graph_from_neo4j_data first.")
        
        # Filter nodes if specified
        if filter_by_type:
            filtered_nodes = [n for n in self.graph.nodes() 
                            if self.graph.nodes[n].get('type') in filter_by_type]
            subgraph = self.graph.subgraph(filtered_nodes)
            pos = {n: self.pos[n] for n in filtered_nodes if n in self.pos}
        else:
            subgraph = self.graph
            pos = self.pos
        
        # Prepare node data
        node_types = [subgraph.nodes[node].get('type', 'Unknown') for node in subgraph.nodes()]
        unique_types = list(set(node_types))
        color_map = self._generate_color_scheme(unique_types)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge info for hover
            edge_data = subgraph.edges[edge]
            rel_type = edge_data.get('rel_type', 'relates_to')
            confidence = edge_data.get('confidence', 1.0)
            edge_info.append(f"{edge[0]} -[{rel_type}]-> {edge[1]} (conf: {confidence:.2f})")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(150,150,150,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces by type
        node_traces = []
        for node_type in unique_types:
            nodes_of_type = [n for n in subgraph.nodes() if subgraph.nodes[n].get('type') == node_type]
            
            node_x = [pos[node][0] for node in nodes_of_type]
            node_y = [pos[node][1] for node in nodes_of_type]
            
            # Node sizes based on degree
            node_sizes = [subgraph.degree(node) * self.config.node_size_multiplier for node in nodes_of_type]
            
            # Node colors
            node_colors = [color_map[node_type] for _ in nodes_of_type]
            
            # Highlight specific nodes if requested
            if highlight_nodes:
                for i, node in enumerate(nodes_of_type):
                    if node in highlight_nodes:
                        node_colors[i] = 'red'
                        node_sizes[i] *= 1.5
            
            # Node hover info
            node_text = []
            for node in nodes_of_type:
                node_data = subgraph.nodes[node]
                info = f"<b>{node}</b><br>"
                info += f"Type: {node_data.get('type', 'Unknown')}<br>"
                info += f"Connections: {subgraph.degree(node)}<br>"
                
                # Add custom properties
                for key, value in node_data.items():
                    if key not in ['type', 'name']:
                        info += f"{key}: {value}<br>"
                
                node_text.append(info)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if self.config.show_labels else 'markers',
                hoverinfo='text',
                text=[node if self.config.show_labels else '' for node in nodes_of_type],
                textposition="middle center",
                hovertext=node_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                name=f"{node_type} ({len(nodes_of_type)})",
                showlegend=True
            )
            
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=[edge_trace] + node_traces,
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents degree centrality",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def visualize_communities(self, communities: Dict[str, int], 
                            title: str = "Community Detection Results") -> go.Figure:
        """
        Visualize detected communities with different colors.
        
        Args:
            communities: Dictionary mapping node names to community IDs
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if not self.graph or not self.pos:
            raise ValueError("Graph not loaded")
        
        # Get unique communities
        unique_communities = list(set(communities.values()))
        community_colors = self._generate_color_scheme([f"Community {i}" for i in unique_communities])
        
        # Create edge trace
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(150,150,150,0.3)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces by community
        community_traces = []
        for comm_id in unique_communities:
            nodes_in_community = [n for n, c in communities.items() if c == comm_id]
            
            if not nodes_in_community:
                continue
            
            node_x = [self.pos[node][0] for node in nodes_in_community if node in self.pos]
            node_y = [self.pos[node][1] for node in nodes_in_community if node in self.pos]
            
            # Node info
            node_text = []
            node_types = []
            for node in nodes_in_community:
                if node not in self.pos:
                    continue
                node_data = self.graph.nodes[node]
                node_type = node_data.get('type', 'Unknown')
                node_types.append(node_type)
                
                info = f"<b>{node}</b><br>"
                info += f"Type: {node_type}<br>"
                info += f"Community: {comm_id}<br>"
                info += f"Connections: {self.graph.degree(node)}"
                node_text.append(info)
            
            # Count types in this community
            type_counts = Counter(node_types)
            dominant_type = type_counts.most_common(1)[0][0] if type_counts else "Mixed"
            
            community_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    size=8,
                    color=community_colors[f"Community {comm_id}"],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f"Community {comm_id} ({len(nodes_in_community)} nodes, {dominant_type})",
                showlegend=True
            )
            
            community_traces.append(community_trace)
        
        # Create figure
        fig = go.Figure(data=[edge_trace] + community_traces,
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_centrality_dashboard(self, centrality_scores: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Create a dashboard showing different centrality measures.
        
        Args:
            centrality_scores: Dictionary mapping node names to centrality measures
            
        Returns:
            Plotly subplot figure
        """
        if not centrality_scores:
            raise ValueError("No centrality scores provided")
        
        # Prepare data
        nodes = list(centrality_scores.keys())
        degree_centrality = [centrality_scores[n].get('degree_centrality', 0) for n in nodes]
        betweenness_centrality = [centrality_scores[n].get('betweenness_centrality', 0) for n in nodes]
        eigenvector_centrality = [centrality_scores[n].get('eigenvector_centrality', 0) for n in nodes]
        
        # Get node types for coloring
        node_types = []
        for node in nodes:
            if self.graph and node in self.graph.nodes:
                node_types.append(self.graph.nodes[node].get('type', 'Unknown'))
            else:
                node_types.append('Unknown')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Degree Centrality', 'Betweenness Centrality', 
                          'Eigenvector Centrality', 'Centrality Comparison'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Degree centrality
        fig.add_trace(
            go.Scatter(
                x=list(range(len(nodes))),
                y=degree_centrality,
                mode='markers',
                marker=dict(size=8, color='blue'),
                text=[f"{node}<br>Type: {node_type}" for node, node_type in zip(nodes, node_types)],
                hoverinfo='text',
                name='Degree'
            ),
            row=1, col=1
        )
        
        # Betweenness centrality
        fig.add_trace(
            go.Scatter(
                x=list(range(len(nodes))),
                y=betweenness_centrality,
                mode='markers',
                marker=dict(size=8, color='red'),
                text=[f"{node}<br>Type: {node_type}" for node, node_type in zip(nodes, node_types)],
                hoverinfo='text',
                name='Betweenness'
            ),
            row=1, col=2
        )
        
        # Eigenvector centrality
        fig.add_trace(
            go.Scatter(
                x=list(range(len(nodes))),
                y=eigenvector_centrality,
                mode='markers',
                marker=dict(size=8, color='green'),
                text=[f"{node}<br>Type: {node_type}" for node, node_type in zip(nodes, node_types)],
                hoverinfo='text',
                name='Eigenvector'
            ),
            row=2, col=1
        )
        
        # Centrality comparison (scatter plot)
        fig.add_trace(
            go.Scatter(
                x=degree_centrality,
                y=betweenness_centrality,
                mode='markers',
                marker=dict(
                    size=[e*50 for e in eigenvector_centrality],
                    color=eigenvector_centrality,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Eigenvector Centrality")
                ),
                text=[f"{node}<br>Type: {node_type}" for node, node_type in zip(nodes, node_types)],
                hoverinfo='text',
                name='Comparison'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Centrality Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        fig.update_xaxes(title_text="Node Index", row=1, col=1)
        fig.update_xaxes(title_text="Node Index", row=1, col=2)
        fig.update_xaxes(title_text="Node Index", row=2, col=1)
        fig.update_xaxes(title_text="Degree Centrality", row=2, col=2)
        
        fig.update_yaxes(title_text="Degree Centrality", row=1, col=1)
        fig.update_yaxes(title_text="Betweenness Centrality", row=1, col=2)
        fig.update_yaxes(title_text="Eigenvector Centrality", row=2, col=1)
        fig.update_yaxes(title_text="Betweenness Centrality", row=2, col=2)
        
        return fig
    
    def create_sector_analysis(self, sector_data: Dict[str, List[str]]) -> go.Figure:
        """
        Create sector-based analysis visualization.
        
        Args:
            sector_data: Dictionary mapping sectors to company lists
            
        Returns:
            Plotly figure with sector analysis
        """
        # Create sector distribution
        sectors = list(sector_data.keys())
        sector_sizes = [len(companies) for companies in sector_data.values()]
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=sectors,
                values=sector_sizes,
                hole=0.3,
                textinfo='label+percent',
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Portfolio Distribution by Sector",
            annotations=[dict(text='Sectors', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str, format: str = "html"):
        """Save visualization to file"""
        try:
            if format.lower() == "html":
                fig.write_html(filename)
            elif format.lower() == "png":
                fig.write_image(filename)
            elif format.lower() == "pdf":
                fig.write_image(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Saved visualization to {filename}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")

def main():
    """Demo the visualization system"""
    visualizer = GraphVisualizer()
    
    # Create sample data
    sample_nodes = [
        {"name": "Acme Security", "type": "Company"},
        {"name": "TechCorp", "type": "Company"},
        {"name": "Machine Learning", "type": "Technology"},
        {"name": "Cybersecurity", "type": "Market"},
        {"name": "CrowdStrike", "type": "Company"}
    ]
    
    sample_edges = [
        {"source": "Acme Security", "target": "Machine Learning", "rel_type": "uses_technology"},
        {"source": "Acme Security", "target": "Cybersecurity", "rel_type": "targets_market"},
        {"source": "Acme Security", "target": "CrowdStrike", "rel_type": "competes_with"},
        {"source": "TechCorp", "target": "Machine Learning", "rel_type": "uses_technology"}
    ]
    
    # Load data and create visualization
    visualizer.load_graph_from_neo4j_data(sample_nodes, sample_edges)
    
    # Create interactive network
    fig = visualizer.create_interactive_network("Demo VC Knowledge Graph")
    
    # Save visualization
    visualizer.save_visualization(fig, "demo_graph.html")
    
    print("Demo visualization created: demo_graph.html")

if __name__ == "__main__":
    main() 