#!/usr/bin/env python3
"""
Graph Visualization System

This module implements visualization capabilities from Section 9 of the strategic 
framework for interactive graph exploration and analysis.

Author: AI Mapping Knowledge Graph System
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphVisualizer:
    """
    Graph visualization system for VC knowledge graphs.
    
    Implements Section 9 framework requirements:
    - Interactive network visualization
    - Community highlighting
    - Centrality-based sizing
    - Export capabilities
    """
    
    def __init__(self):
        self.graph = None
        self.pos = None
    
    def load_networkx_graph(self, graph: nx.Graph):
        """Load a NetworkX graph for visualization"""
        self.graph = graph
        self.pos = nx.spring_layout(graph, k=1, iterations=50)
        logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    def create_interactive_plot(self, title: str = "Knowledge Graph", 
                              color_by: str = "type",
                              size_by: str = "degree") -> go.Figure:
        """
        Create an interactive Plotly visualization.
        
        Args:
            title: Chart title
            color_by: Node attribute to color by
            size_by: Node attribute to size by ('degree' or node attribute)
            
        Returns:
            Plotly figure object
        """
        if not self.graph:
            raise ValueError("No graph loaded. Call load_networkx_graph first.")
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node data
        node_x = [self.pos[node][0] for node in self.graph.nodes()]
        node_y = [self.pos[node][1] for node in self.graph.nodes()]
        
        # Node colors
        if color_by == "type":
            node_colors = [self.graph.nodes[node].get('type', 'Unknown') for node in self.graph.nodes()]
        else:
            node_colors = [self.graph.nodes[node].get(color_by, 0) for node in self.graph.nodes()]
        
        # Node sizes
        if size_by == "degree":
            node_sizes = [self.graph.degree(node) * 5 for node in self.graph.nodes()]
        else:
            node_sizes = [self.graph.nodes[node].get(size_by, 5) * 5 for node in self.graph.nodes()]
        
        # Node hover text
        node_text = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            text = f"<b>{node}</b><br>"
            text += f"Type: {node_data.get('type', 'Unknown')}<br>"
            text += f"Connections: {self.graph.degree(node)}"
            node_text.append(text)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Set3' if color_by == "type" else 'Viridis',
                line=dict(width=1, color='white'),
                opacity=0.8
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
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
    
    def create_matplotlib_plot(self, title: str = "Knowledge Graph", 
                             figsize: tuple = (12, 8),
                             node_color_map: Optional[Dict] = None) -> plt.Figure:
        """
        Create a static matplotlib visualization.
        
        Args:
            title: Chart title
            figsize: Figure size
            node_color_map: Mapping of node types to colors
            
        Returns:
            Matplotlib figure object
        """
        if not self.graph:
            raise ValueError("No graph loaded")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Default color map
        if not node_color_map:
            node_types = set(self.graph.nodes[node].get('type', 'Unknown') for node in self.graph.nodes())
            colors = plt.cm.Set3(range(len(node_types)))
            node_color_map = dict(zip(node_types, colors))
        
        # Node colors and sizes
        node_colors = [node_color_map.get(self.graph.nodes[node].get('type', 'Unknown'), 'gray') 
                      for node in self.graph.nodes()]
        node_sizes = [self.graph.degree(node) * 50 for node in self.graph.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, self.pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.8, ax=ax)
        
        nx.draw_networkx_edges(self.graph, self.pos, 
                              alpha=0.3, 
                              width=1, ax=ax)
        
        nx.draw_networkx_labels(self.graph, self.pos, 
                               font_size=8, ax=ax)
        
        ax.set_title(title, size=16)
        ax.axis('off')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=node_type)
                          for node_type, color in node_color_map.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig
    
    def save_interactive_html(self, fig: go.Figure, filename: str):
        """Save interactive visualization as HTML"""
        try:
            fig.write_html(filename)
            logger.info(f"Saved interactive visualization to {filename}")
        except Exception as e:
            logger.error(f"Error saving HTML: {e}")
    
    def save_static_image(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """Save static visualization as image"""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved static visualization to {filename}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")

def main():
    """Demo the visualization system"""
    visualizer = GraphVisualizer()
    
    # Create sample graph
    G = nx.Graph()
    G.add_node("Acme Security", type="Company")
    G.add_node("Machine Learning", type="Technology") 
    G.add_node("Cybersecurity", type="Market")
    G.add_node("CrowdStrike", type="Company")
    
    G.add_edge("Acme Security", "Machine Learning")
    G.add_edge("Acme Security", "Cybersecurity")
    G.add_edge("Acme Security", "CrowdStrike")
    
    # Load and visualize
    visualizer.load_networkx_graph(G)
    
    # Create interactive plot
    interactive_fig = visualizer.create_interactive_plot("Demo VC Knowledge Graph")
    visualizer.save_interactive_html(interactive_fig, "demo_interactive.html")
    
    # Create static plot
    static_fig = visualizer.create_matplotlib_plot("Demo VC Knowledge Graph")
    visualizer.save_static_image(static_fig, "demo_static.png")
    
    print("Demo visualizations created!")

if __name__ == "__main__":
    main() 