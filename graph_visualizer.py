#!/usr/bin/env python3
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
from typing import Dict, Any, Optional
import logging
from functools import lru_cache
import community as community_louvain

# Pre-defined styling for different node and edge types
NODE_STYLES: Dict[str, Dict[str, Any]] = {
    "Company": {"shape": "circle", "color_scale": px.colors.sequential.Blues, "base_size": 20},
    "Technology": {"shape": "square", "color_scale": px.colors.sequential.Greens, "base_size": 15},
    "Market": {"shape": "diamond", "color_scale": px.colors.sequential.Oranges, "base_size": 18},
    "Investor": {"shape": "triangle-up", "color_scale": px.colors.sequential.Purples, "base_size": 16},
}

EDGE_STYLES: Dict[str, Dict[str, Any]] = {
    "direct": {"dash": "solid", "width": 2, "opacity": 0.8},
    "inferred": {"dash": "dash", "width": 1, "opacity": 0.5},
    "strong": {"dash": "solid", "width": 3, "opacity": 1.0},
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Graph visualization system for VC knowledge graphs.

    Implements Section 9 framework requirements:
    - Interactive network visualization
    - Community highlighting with Louvain detection
    - Centrality-based sizing (degree, betweenness, pagerank)
    - Distinct shapes for node types
    - Differentiated edge styles
    - Export capabilities
    """

    def __init__(self):
        self.graph = None
        self.pos = None

    def load_networkx_graph(self, graph: nx.Graph):
        """Load a NetworkX graph for visualization"""
        self.graph = graph
        self.pos = nx.spring_layout(graph, k=1, iterations=50)
        logger.info(
            f"Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        # Clear caches when a new graph is loaded
        self._get_centrality.cache_clear()
        self._get_communities.cache_clear()

    @lru_cache(maxsize=None)
    def _get_centrality(self, measure: str) -> Dict[Any, float]:
        """Compute and cache centrality measures."""
        if measure == "degree":
            values = dict(self.graph.degree())
        elif measure == "betweenness":
            values = nx.betweenness_centrality(self.graph)
        elif measure == "pagerank":
            values = nx.pagerank(self.graph)
        else:
            values = {n: self.graph.nodes[n].get(measure, 1) for n in self.graph.nodes()}
        return values

    @lru_cache(maxsize=None)
    def _get_communities(self) -> Dict[Any, int]:
        """Detect communities using Louvain algorithm and cache the result."""
        return community_louvain.best_partition(self.graph)

    def create_interactive_plot(
        self,
        title: str = "Knowledge Graph",
        layout_algorithm: str = "force_directed",
        centrality_measure: str = "pagerank",
        search: Optional[str] = None,
        theme: str = "light",
    ) -> go.Figure:
        """Create a rich interactive Plotly visualization with advanced features."""

        if not self.graph:
            raise ValueError("No graph loaded. Call load_networkx_graph first.")

        # Choose layout algorithm
        if layout_algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout_algorithm == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        self.pos = pos

        # Centrality-based sizing
        centrality = self._get_centrality(centrality_measure)

        # Community detection for coloring
        communities = self._get_communities()

        edge_traces = []
        for u, v, data in self.graph.edges(data=True):
            style = EDGE_STYLES["inferred"] if data.get("inferred") else EDGE_STYLES["direct"]
            if data.get("confidence", 1.0) > 0.8:
                style = EDGE_STYLES.get("strong", style)

            edge_traces.append(
                go.Scatter(
                    x=[pos[u][0], pos[v][0]],
                    y=[pos[u][1], pos[v][1]],
                    mode="lines",
                    line=dict(color="gray", dash=style["dash"], width=style["width"], opacity=style["opacity"]),
                    hoverinfo="text",
                    hovertext=f"{u} → {v} ({data.get('rel_type','rel')})",
                    showlegend=False,
                )
            )

        node_traces = []
        node_types = nx.get_node_attributes(self.graph, "type")
        for node_type, style in NODE_STYLES.items():
            nodes_of_type = [n for n, t in node_types.items() if t == node_type]
            if not nodes_of_type:
                continue

            colors = []
            sizes = []
            texts = []

            color_scale = style["color_scale"]
            for n in nodes_of_type:
                com_id = communities.get(n, 0)
                color = color_scale[com_id % len(color_scale)]
                if search and search.lower() in n.lower():
                    color = "#FF0000"
                colors.append(color)
                size = style["base_size"] * (1 + centrality.get(n, 0.1))
                sizes.append(size)

                data = self.graph.nodes[n]
                text = f"<b>{n}</b><br>Type: {node_type}"
                for k, v in data.items():
                    if k not in ["type", "name"]:
                        text += f"<br>{k}: {v}"
                texts.append(text)

            node_traces.append(
                go.Scatter(
                    x=[pos[n][0] for n in nodes_of_type],
                    y=[pos[n][1] for n in nodes_of_type],
                    mode="markers",
                    marker=dict(size=sizes, color=colors, symbol=style["shape"], line=dict(width=1, color="white")),
                    hoverinfo="text",
                    hovertext=texts,
                    name=node_type,
                )
            )

        layout_bg = "#FFFFFF" if theme == "light" else "#1e1e1e"
        font_color = "#000" if theme == "light" else "#EEE"

        fig = go.Figure(data=edge_traces + node_traces)
        fig.update_layout(
            title=title,
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=layout_bg,
            paper_bgcolor=layout_bg,
            font=dict(color=font_color),
        )

        return fig

    def create_matplotlib_plot(
        self,
        title: str = "Knowledge Graph",
        figsize: tuple = (12, 8),
        node_color_map: Optional[Dict] = None,
    ) -> plt.Figure:
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
            node_types = set(
                self.graph.nodes[node].get("type", "Unknown") for node in self.graph.nodes()
            )
            colors = plt.cm.Set3(range(len(node_types)))
            node_color_map = dict(zip(node_types, colors))

        # Node colors and sizes
        node_colors = [
            node_color_map.get(self.graph.nodes[node].get("type", "Unknown"), "gray")
            for node in self.graph.nodes()
        ]
        node_sizes = [self.graph.degree(node) * 50 for node in self.graph.nodes()]

        # Draw the graph
        nx.draw_networkx_nodes(
            self.graph, self.pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
        )

        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.3, width=1, ax=ax)

        nx.draw_networkx_labels(self.graph, self.pos, font_size=8, ax=ax)

        ax.set_title(title, size=16)
        ax.axis("off")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=node_type,
            )
            for node_type, color in node_color_map.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        return fig

    def export_figure(self, fig: go.Figure, filename: str) -> None:
        """Export interactive figure to PNG, SVG, or JSON based on filename."""
        try:
            if filename.lower().endswith(".png"):
                fig.write_image(filename, format="png")
            elif filename.lower().endswith(".svg"):
                fig.write_image(filename, format="svg")
            elif filename.lower().endswith(".json"):
                fig.write_json(filename)
            elif filename.lower().endswith(".html"):
                fig.write_html(filename)
            else:
                raise ValueError("Unsupported export format")
            logger.info(f"Exported figure to {filename}")
        except Exception as e:
            logger.error(f"Error exporting figure: {e}")

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
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved static visualization to {filename}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")

