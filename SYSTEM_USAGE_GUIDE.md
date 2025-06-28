# VC Intelligence Knowledge Graph System - Complete Usage Guide

## 🎯 Strategic Framework Implementation

This system implements the complete **"Strategic Framework for Venture Capital Intelligence: Leveraging Knowledge Graphs for Meta-Analysis and Opportunity Discovery"** as outlined in the strategic framework document.

## 📋 System Overview

The VC Intelligence Knowledge Graph System transforms unstructured investment data into actionable intelligence through:

- **Knowledge Extraction**: Convert CSV data into structured Subject-Predicate-Object triplets
- **VC Ontology**: Purpose-built ontology for venture capital analysis
- **Meta-Graph Architecture**: Unified knowledge graph combining multiple data sources  
- **Advanced Analytics**: Community detection, centrality analysis, link prediction
- **Natural Language Interface**: Query the graph using business questions
- **Visualization System**: Interactive and static graph visualizations
- **Evaluation Framework**: Quality control and continuous improvement

## 🚀 Quick Start

### 1. System Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test system setup
python setup_system.py

# Run comprehensive system test
python complete_system_test.py
```

### 2. Basic Demo

```bash
# Run the demonstration
python demo_system.py
```

This will show:
- Knowledge extraction from AI mapping data
- VC ontology mapping examples
- Analytical insights (sector analysis, autonomy analysis, technology patterns)
- Competitive landscape analysis
- White space opportunity identification
- Investment thesis pattern recognition

### 3. Complete Pipeline

```bash
# Run the full end-to-end pipeline
python main_pipeline.py
```

This executes all 5 phases:
1. Knowledge extraction from CSV
2. Meta-graph creation in Neo4j
3. Advanced analytics
4. Visualization generation
5. Executive summary reporting

## 📊 Core Components

### 1. Knowledge Extraction (`csv_knowledge_extractor.py`)

**Purpose**: Convert CSV data into knowledge graph triplets using VC-specific ontology

**Usage**:
```python
from csv_knowledge_extractor import VCOntologyExtractor

extractor = VCOntologyExtractor()
triplets = extractor.process_csv("raw ai mapping data.csv")
print(f"Extracted {len(triplets)} knowledge triplets")
```

**VC Ontology Entities**:
- **Company**: The startups/firms being analyzed
- **Technology**: Technical approaches, algorithms, platforms
- **Market**: Target markets, customer segments
- **MarketFriction**: Problems, pain points, market gaps
- **Differentiator**: Unique advantages, competitive moats
- **ThesisConcept**: Investment theses, strategic assumptions

**Relationship Types**:
- `operates_in`: Company operates in Market
- `addresses_friction`: Company addresses MarketFriction  
- `has_differentiator`: Company has Differentiator
- `uses_technology`: Company uses Technology
- `targets_market`: Company targets Market
- `competes_with`: Company competes with Company

### 2. Neo4j Integration (`neo4j_ingestion.py`)

**Purpose**: Create unified meta-graph by merging individual knowledge graphs

**Setup Neo4j**:
1. Install Neo4j Desktop or server
2. Create a new database
3. Set credentials (default: neo4j/password)
4. Enable APOC plugin for JSON import

**Usage**:
```python
from neo4j_ingestion import Neo4jIngestion

neo4j = Neo4jIngestion("bolt://localhost:7687", "neo4j", "password")
neo4j.connect()
neo4j.create_indexes()
neo4j.ingest_triplets_from_json("triplets.json")
```

### 3. Advanced Analytics (`graph_analytics.py`)

**Purpose**: Extract strategic insights using graph algorithms

**Community Detection** (Section 4):
```python
from graph_analytics import GraphAnalytics

analytics = GraphAnalytics()
analytics.connect_neo4j()
analytics.load_graph_from_neo4j()

# Find thematic clusters
communities = analytics.detect_communities_louvain()
for community in communities:
    print(f"Community {community.community_id}: {community.theme_description}")
```

**Centrality Analysis** (Section 6):
```python
# Identify keystone technologies and influential nodes
centrality_scores = analytics.calculate_centrality_measures()
keystone_tech = analytics.analyze_keystone_technologies(centrality_scores)
```

**Link Prediction** (Section 7):
```python
# Discover white space opportunities
predictions = analytics.predict_links_adamic_adar()
high_value_ops = [p for p in predictions if p.prediction_score > 0.1]
```

**Subgraph Comparison** (Section 5):
```python
# Compare investment theses
comparison = analytics.compare_company_subgraphs("CompanyA", "CompanyB")
print(f"Shared connections: {comparison.shared_nodes}")
```

### 4. Natural Language Queries (`knowledge_query_interface.py`)

**Purpose**: Query the knowledge graph using business questions

**Usage**:
```python
from knowledge_query_interface import KnowledgeQueryInterface

query_interface = KnowledgeQueryInterface()

# Ask business questions
result = query_interface.process_natural_language_query(
    "What technologies does Palantir use?"
)

print(result.interpretation)
for insight in result.insights:
    print(f"💡 {insight}")
```

**Sample Queries**:
- "Who competes with [Company]?"
- "What technologies does [Company] use?"
- "Which companies target the cybersecurity market?"
- "What problems does [Company] solve?"
- "What makes [Company] different?"
- "What are the gaps in the AI security space?"

### 5. Visualization (`visualization_system.py`)

**Purpose**: Create interactive and static visualizations

**Interactive Network**:
```python
from visualization_system import VisualizationSystem

visualizer = VisualizationSystem()
visualizer.create_interactive_network(
    graph_data, 
    output_file="network.html"
)
```

**Static Analysis Plots**:
```python
# Create sector analysis plot
visualizer.create_sector_analysis_plot(data, "sector_analysis.png")

# Create technology landscape plot  
visualizer.create_technology_landscape(data, "tech_landscape.png")
```

### 6. Evaluation Framework (`evaluation_framework.py`)

**Purpose**: Quality control and continuous improvement

**Create Golden Dataset**:
```python
from evaluation_framework import EvaluationFramework

evaluator = EvaluationFramework()

# Manually annotate sample documents
golden_dataset = evaluator.create_golden_dataset(
    documents=sample_docs,
    output_file="golden_standard.json"
)
```

**Evaluate Performance**:
```python
# Calculate metrics
metrics = evaluator.calculate_classical_metrics(
    predicted_triplets, 
    ground_truth_triplets
)

print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"F1-Score: {metrics.f1_score:.3f}")

# Analyze errors for improvement
error_analysis = evaluator.perform_error_analysis(
    predicted_triplets, 
    ground_truth_triplets
)
```

## 🎯 Use Cases & Examples

### Investment Portfolio Analysis

```python
# Analyze entire portfolio for thematic patterns
pipeline = VCIntelligencePipeline()
results = pipeline.run_complete_pipeline("portfolio_data.csv")

print(f"Found {len(results['communities'])} investment themes:")
for theme in results['communities']:
    print(f"- {theme['theme']}: {theme['size']} companies")
```

### Competitive Intelligence

```python
# Compare competitive positioning
analytics = GraphAnalytics()
comparison = analytics.compare_company_subgraphs("StartupA", "StartupB")

print(f"Shared markets: {comparison.shared_markets}")
print(f"Technology overlap: {comparison.shared_technologies}")
print(f"Competitive alignment: {comparison.similarity_score:.2f}")
```

### White Space Discovery

```python
# Find underserved market opportunities
predictions = analytics.predict_links_adamic_adar(node_type_filter="Market")
opportunities = [p for p in predictions if p.prediction_score > 0.15]

print("Top white space opportunities:")
for opp in opportunities[:5]:
    print(f"- {opp.node1} -> {opp.node2} (Score: {opp.prediction_score:.3f})")
```

### Market Research

```python
# Query market landscape
query_interface = KnowledgeQueryInterface()

# Find all companies in a sector
result = query_interface.process_natural_language_query(
    "Which companies operate in cybersecurity?"
)

# Analyze market friction
friction_result = query_interface.process_natural_language_query(
    "What problems do cybersecurity companies solve?"
)
```

## 📈 Advanced Analytics Workflows

### 1. Thematic Investment Analysis

```python
# Complete thematic analysis workflow
def analyze_investment_themes():
    analytics = GraphAnalytics()
    analytics.connect_neo4j()
    analytics.load_graph_from_neo4j()
    
    # Detect communities
    communities = analytics.detect_communities_louvain()
    
    # Analyze each theme
    for community in communities:
        # Get community details
        companies = analytics.get_community_companies(community.community_id)
        technologies = analytics.get_community_technologies(community.community_id)
        
        # Calculate theme strength
        centrality = analytics.calculate_community_centrality(community.community_id)
        
        # Generate investment thesis
        thesis = analytics.generate_theme_thesis(community)
        
        print(f"\n🎯 Investment Theme: {community.theme_description}")
        print(f"   Companies: {len(companies)}")
        print(f"   Key Technologies: {technologies[:3]}")
        print(f"   Theme Strength: {centrality:.3f}")
        print(f"   Investment Thesis: {thesis}")
```

### 2. Risk Analysis

```python
# Identify systemic risks and dependencies
def analyze_portfolio_risks():
    analytics = GraphAnalytics()
    
    # Find high-centrality nodes (potential points of failure)
    centrality_scores = analytics.calculate_centrality_measures()
    high_risk_nodes = [n for n in centrality_scores if n.betweenness_centrality > 0.1]
    
    # Analyze thesis concentration
    thesis_analysis = analytics.analyze_thesis_concentration()
    
    # Identify market dependencies
    market_deps = analytics.analyze_market_dependencies()
    
    return {
        'systemic_risks': high_risk_nodes,
        'thesis_concentration': thesis_analysis,
        'market_dependencies': market_deps
    }
```

### 3. Due Diligence Support

```python
# Comprehensive company analysis for due diligence
def analyze_company_for_dd(company_name):
    analytics = GraphAnalytics()
    query_interface = KnowledgeQueryInterface()
    
    # Get company subgraph
    subgraph = analytics.get_company_subgraph(company_name)
    
    # Find competitors
    competitors = query_interface.process_natural_language_query(
        f"Who competes with {company_name}?"
    )
    
    # Analyze differentiation
    differentiators = query_interface.process_natural_language_query(
        f"What makes {company_name} different?"
    )
    
    # Market positioning
    markets = query_interface.process_natural_language_query(
        f"What markets does {company_name} target?"
    )
    
    # Technology analysis
    technologies = query_interface.process_natural_language_query(
        f"What technologies does {company_name} use?"
    )
    
    return {
        'company': company_name,
        'competitive_landscape': competitors.results,
        'key_differentiators': differentiators.results,
        'target_markets': markets.results,
        'technology_stack': technologies.results,
        'subgraph_analysis': subgraph
    }
```

## 🔧 Configuration & Customization

### System Configuration (`system_config.json`)

```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j", 
    "password": "password"
  },
  "pipeline": {
    "batch_size": 1000,
    "output_directory": "output",
    "enable_visualization": true,
    "enable_evaluation": true
  },
  "analytics": {
    "community_detection": {
      "algorithm": "louvain",
      "resolution": 1.0
    },
    "centrality": {
      "calculate_all": true,
      "top_n_nodes": 20
    },
    "link_prediction": {
      "method": "adamic_adar", 
      "threshold": 0.1,
      "max_predictions": 100
    }
  }
}
```

### Custom Ontology Extension

To add new entity types or relationships:

1. **Extend the ontology** in `csv_knowledge_extractor.py`:
```python
# Add new entity type
def extract_funding_info(self, row: Dict[str, Any]) -> List[KnowledgeTriplet]:
    """Extract funding information"""
    triplets = []
    
    if 'Funding_Round' in row:
        triplets.append(KnowledgeTriplet(
            subject=company_name,
            predicate="completed_funding",
            object=row['Funding_Round'],
            subject_type="Company",
            object_type="FundingEvent"
        ))
    
    return triplets
```

2. **Update Neo4j ingestion** to handle new entity types

3. **Extend analytics** to work with new relationships

### Custom Analytics

Add domain-specific analytics:

```python
class CustomVCAnalytics(GraphAnalytics):
    
    def analyze_funding_patterns(self):
        """Analyze funding round patterns"""
        query = """
        MATCH (c:Entity {type: 'Company'})-[:RELATES_TO]->(f:Entity {type: 'FundingEvent'})
        RETURN c.name as company, f.name as funding_round, count(*) as frequency
        """
        results = self.run_neo4j_query(query)
        return self.process_funding_analysis(results)
    
    def identify_emerging_sectors(self):
        """Identify rapidly growing sectors"""
        # Custom sector growth analysis
        pass
```

## 📊 Performance & Optimization

### Batch Processing

For large datasets:

```python
# Process data in batches
extractor = VCOntologyExtractor()
batch_size = 100

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    triplets = extractor.process_batch(batch)
    save_batch_results(triplets, f"batch_{i//batch_size}.json")
```

### Neo4j Optimization

```cypher
-- Create indexes for performance
CREATE INDEX FOR (n:Entity) ON (n.name)
CREATE INDEX FOR (n:Entity) ON (n.type)
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.type)

-- Query optimization
MATCH (c:Entity {type: 'Company'})-[r:RELATES_TO]->(t:Entity {type: 'Technology'})
WHERE c.name = 'CompanyName'
RETURN t.name
```

### Memory Management

```python
# For large graphs, use subgraph sampling
def analyze_large_graph():
    analytics = GraphAnalytics()
    
    # Sample subgraph for analysis
    subgraph = analytics.sample_subgraph(sample_size=1000)
    
    # Run analytics on sample
    communities = analytics.detect_communities_louvain(graph=subgraph)
    
    return communities
```

## 🚨 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   ```bash
   # Check Neo4j is running
   sudo systemctl status neo4j
   
   # Start Neo4j
   sudo systemctl start neo4j
   ```

2. **Memory Issues with Large Graphs**
   ```python
   # Increase memory allocation
   import os
   os.environ['JAVA_OPTS'] = '-Xmx8G'
   ```

3. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install -r requirements.txt
   
   # Verify installation
   python setup_system.py
   ```

4. **Data Format Issues**
   ```python
   # Check CSV data format
   import pandas as pd
   df = pd.read_csv("data.csv")
   print(df.columns.tolist())
   print(df.head())
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
extractor = VCOntologyExtractor(debug=True)
```

## 📚 Further Reading

- **Strategic Framework Document**: Complete theoretical foundation
- **Neo4j Documentation**: https://neo4j.com/docs/
- **NetworkX Documentation**: https://networkx.org/documentation/
- **Graph Theory Concepts**: Community detection, centrality measures
- **Knowledge Graph Best Practices**: Ontology design, data modeling

## 🤝 Support & Contribution

For issues or enhancements:

1. **System Testing**: Run `python complete_system_test.py`
2. **Performance Profiling**: Use built-in timing and metrics
3. **Customization**: Extend ontology and analytics as needed
4. **Integration**: Connect with existing VC tools and databases

## 🎉 Success Metrics

The system is successful when it provides:

- **Faster Due Diligence**: Reduced time to analyze company positioning
- **Better Pattern Recognition**: Discovery of non-obvious investment themes  
- **Risk Mitigation**: Early identification of portfolio concentration risks
- **Opportunity Discovery**: Data-driven identification of white space
- **Competitive Intelligence**: Comprehensive competitive landscape analysis
- **Investment Thesis Validation**: Evidence-based thesis development

---

**Ready to transform your VC intelligence capabilities? Start with the demo and explore the endless possibilities of knowledge graph-powered investment analysis.** 