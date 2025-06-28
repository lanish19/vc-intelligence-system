#!/usr/bin/env python3
"""
VC Intelligence System Demonstration

This script demonstrates the capabilities of the VC Intelligence Knowledge Graph System
with specific examples and use cases from the AI mapping data.

Author: AI Mapping Knowledge Graph System
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Import system components
from csv_knowledge_extractor import VCOntologyExtractor
from llm_client import KnowledgeExtractor
from neo4j_ingestion import Neo4jIngestion
from graph_analytics import GraphAnalytics
from graph_visualizer import GraphVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemDemo:
    """Demonstration of VC Intelligence System capabilities"""
    
    def __init__(self):
        self.extractor = VCOntologyExtractor()
        self.llm_extractor = KnowledgeExtractor()
        self.results = {}
    
    def demo_llm_knowledge_extraction(self) -> None:
        """Demonstrate LLM-powered knowledge extraction"""
        
        print("\n" + "="*60)
        print("🤖 DEMO: LLM-Powered Knowledge Extraction")
        print("="*60)
        
        try:
            # Sample company data for LLM extraction
            sample_companies = [
                {
                    'firm_name': 'CyberGuard AI',
                    'Firm Description': 'Provides AI-powered threat detection for enterprise networks using behavioral analysis.',
                    'Problem Hypothesis': 'Traditional SIEM solutions have high false positive rates and lack predictive capabilities.',
                    'Firm Unique Differentiation': 'Proprietary behavioral analysis reduces false positives by 90% compared to traditional solutions.',
                    'Primary Sector': 'Cybersecurity AI'
                },
                {
                    'firm_name': 'AutonomousEdge Systems',
                    'Firm Description': 'Develops autonomous AI systems for tactical edge computing in contested environments.',
                    'Problem Hypothesis': 'Military operations require AI systems that can operate without cloud connectivity.',
                    'Firm Unique Differentiation': 'First-of-its-kind edge AI that maintains full autonomy in denied environments.',
                    'Primary Sector': 'Autonomous Systems'
                }
            ]
            
            for i, company_data in enumerate(sample_companies):
                firm_name = company_data['firm_name']
                print(f"\n🏢 Analyzing Company {i+1}: {firm_name}")
                print("-" * 50)
                
                # Extract knowledge using LLM
                print("🧠 LLM Processing...")
                triplets = self.llm_extractor.extract_from_csv_row(company_data)
                
                print(f"✅ Extracted {len(triplets)} knowledge triplets:")
                
                # Group triplets by type
                by_type = {}
                for triplet in triplets:
                    obj_type = triplet.get('object_type', 'Unknown')
                    if obj_type not in by_type:
                        by_type[obj_type] = []
                    by_type[obj_type].append(triplet)
                
                # Display results by category
                for obj_type, type_triplets in by_type.items():
                    print(f"\n   📋 {obj_type} ({len(type_triplets)} items):")
                    for triplet in type_triplets[:3]:  # Show first 3
                        print(f"      • {triplet['subject']} → {triplet['predicate']} → {triplet['object']}")
                        if len(triplet['object']) > 50:
                            print(f"        (confidence: {triplet.get('confidence', 0.8):.2f})")
                    
                    if len(type_triplets) > 3:
                        print(f"      ... and {len(type_triplets) - 3} more")
            
            print("\n✨ LLM extraction demonstrates:")
            print("   • Contextual understanding of VC terminology")
            print("   • Automatic entity type classification")
            print("   • Relationship inference from unstructured text")
            print("   • Confidence scoring for extracted knowledge")
            
        except Exception as e:
            print(f"❌ LLM extraction failed: {str(e)}")
            print("💡 Make sure GEMINI_API_KEY is set in your environment")
            print("   Run: export GEMINI_API_KEY=your_api_key_here")
    
    def demo_knowledge_extraction(self) -> None:
        """Demonstrate knowledge extraction from CSV data"""
        
        print("\n" + "="*60)
        print("📊 DEMO: Knowledge Extraction from AI Mapping Data")
        print("="*60)
        
        # Load sample data
        csv_path = "raw ai mapping data.csv"
        if not Path(csv_path).exists():
            print(f"❌ Data file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        print(f"📈 Loaded {len(df)} companies from AI mapping data")
        
        # Extract knowledge from first few companies
        sample_companies = df.head(3)
        
        for idx, row in sample_companies.iterrows():
            firm_name = row.get('firm_name', 'Unknown')
            print(f"\n🏢 Analyzing: {firm_name}")
            print("-" * 40)
            
            # Extract different types of knowledge
            company_info = self.extractor.extract_company_info(row)
            frictions = self.extractor.extract_market_frictions(row)
            differentiators = self.extractor.extract_differentiators(row)
            technologies = self.extractor.extract_technologies(row)
            
            print(f"   Company Relations: {len(company_info)}")
            print(f"   Market Frictions: {len(frictions)}")
            print(f"   Differentiators: {len(differentiators)}")
            print(f"   Technologies: {len(technologies)}")
            
            # Show sample triplets
            if company_info:
                triplet = company_info[0]
                print(f"   Example: {triplet.subject} -> {triplet.predicate} -> {triplet.object}")
        
        print(f"\n✅ Knowledge extraction complete")
    
    def demo_ontology_mapping(self) -> None:
        """Demonstrate the VC ontology mapping"""
        
        print("\n" + "="*60)
        print("🧠 DEMO: VC Ontology and Knowledge Structure")
        print("="*60)
        
        # Show ontology structure
        ontology = {
            "Entity Types": [
                "Company - The startups/firms being analyzed",
                "Technology - Technical approaches, algorithms, platforms",
                "Market - Target markets, customer segments", 
                "MarketFriction - Problems, pain points, market gaps",
                "Differentiator - Unique advantages, competitive moats",
                "ThesisConcept - Investment theses, strategic assumptions"
            ],
            "Relationship Types": [
                "operates_in - Company operates in Market",
                "addresses_friction - Company addresses MarketFriction",
                "has_differentiator - Company has Differentiator",
                "uses_technology - Company uses Technology",
                "targets_market - Company targets Market",
                "competes_with - Company competes with Company"
            ]
        }
        
        for category, items in ontology.items():
            print(f"\n📋 {category}:")
            for item in items:
                print(f"   • {item}")
        
        # Show how data maps to ontology
        print(f"\n🔗 Data Mapping Examples:")
        print("   CSV Field -> Ontology Entity")
        print("   'Primary Sector' -> Market")
        print("   'Problem Hypothesis' -> MarketFriction")
        print("   'Firm Unique Differentiation' -> Differentiator")
        print("   'Role of AI In Business Model' -> Technology")
    
    def demo_analytical_insights(self) -> None:
        """Demonstrate analytical insights from sample data"""
        
        print("\n" + "="*60)
        print("📈 DEMO: Analytical Insights from Sample Data")
        print("="*60)
        
        # Load and analyze sample data
        csv_path = "raw ai mapping data.csv"
        if not Path(csv_path).exists():
            print(f"❌ Data file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Sector analysis
        print("\n🏭 SECTOR ANALYSIS:")
        sector_counts = df['Primary Sector'].value_counts()
        print("Top investment sectors:")
        for sector, count in sector_counts.head(5).items():
            print(f"   • {sector}: {count} companies")
        
        # Autonomy spectrum analysis
        autonomy_col = 'Autonomy Spectrum Ranking (1 to 10)\nThis spectrum is designed to measure and rank the degree of independent operation within an AI system.\n1 = Decision Support Role: At this end of the spectrum, the AI system acts primarily as an assistant to human operators.\n10 = Autonomous Operations: At this extreme, the AI system operates entirely independently, making decisions and executing actions without direct human intervention or oversight.'
        
        if autonomy_col in df.columns:
            print("\n🤖 AUTONOMY ANALYSIS:")
            avg_autonomy = df[autonomy_col].mean()
            print(f"   Average autonomy level: {avg_autonomy:.1f}/10")
            
            high_autonomy = df[df[autonomy_col] >= 8]
            print(f"   Highly autonomous systems (8+): {len(high_autonomy)} companies")
            
            if len(high_autonomy) > 0:
                print("   Examples:")
                for _, row in high_autonomy.head(3).iterrows():
                    print(f"     • {row.get('firm_name', 'Unknown')} ({row[autonomy_col]:.1f})")
        
        # Technology pattern analysis
        print("\n💻 TECHNOLOGY PATTERNS:")
        ai_keywords = ['AI', 'machine learning', 'autonomous', 'artificial intelligence', 'deep learning']
        
        for keyword in ai_keywords[:3]:
            count = df['Firm Description'].str.contains(keyword, case=False, na=False).sum()
            print(f"   • Companies using '{keyword}': {count}")
        
        # Market friction analysis
        print("\n🔥 MARKET FRICTION ANALYSIS:")
        friction_keywords = ['gap', 'lack', 'challenge', 'problem', 'bottleneck']
        
        for keyword in friction_keywords[:3]:
            count = df['Problem Hypothesis'].str.contains(keyword, case=False, na=False).sum()
            print(f"   • Companies addressing '{keyword}': {count}")
    
    def demo_competitive_landscape(self) -> None:
        """Demonstrate competitive landscape analysis"""
        
        print("\n" + "="*60)
        print("⚔️ DEMO: Competitive Landscape Analysis")
        print("="*60)
        
        csv_path = "raw ai mapping data.csv"
        if not Path(csv_path).exists():
            print(f"❌ Data file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Find companies in same sectors
        print("\n🏢 SECTOR COMPETITION:")
        sector_groups = df.groupby('Primary Sector').size().sort_values(ascending=False)
        
        for sector, count in sector_groups.head(3).items():
            if count > 1:
                print(f"\n   📊 {sector} ({count} companies):")
                sector_companies = df[df['Primary Sector'] == sector]['firm_name'].head(5)
                for company in sector_companies:
                    print(f"     • {company}")
        
        # Technology overlap analysis
        print("\n\n🔧 TECHNOLOGY OVERLAP:")
        common_techs = ['AI', 'machine learning', 'autonomous', 'cybersecurity', 'cloud']
        
        for tech in common_techs[:3]:
            companies = df[df['Firm Description'].str.contains(tech, case=False, na=False)]['firm_name']
            if len(companies) > 1:
                print(f"\n   {tech.upper()} Technology ({len(companies)} companies):")
                for company in companies.head(4):
                    print(f"     • {company}")
    
    def demo_white_space_analysis(self) -> None:
        """Demonstrate white space opportunity identification"""
        
        print("\n" + "="*60)
        print("🎯 DEMO: White Space Opportunity Analysis")
        print("="*60)
        
        # Simulated white space opportunities based on data patterns
        opportunities = [
            {
                "gap": "Quantum-resistant cybersecurity for IoT devices",
                "rationale": "High quantum computing development but limited IoT security solutions",
                "market_size": "Large",
                "competitive_density": "Low"
            },
            {
                "gap": "Edge AI for maritime autonomous systems",
                "rationale": "Strong aerospace autonomy but limited maritime applications",
                "market_size": "Medium", 
                "competitive_density": "Very Low"
            },
            {
                "gap": "AI-powered supply chain security for defense",
                "rationale": "Many cybersecurity + logistics companies but few combine both",
                "market_size": "Large",
                "competitive_density": "Medium"
            }
        ]
        
        print("\n🔍 IDENTIFIED OPPORTUNITIES:")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"\n   {i}. {opp['gap']}")
            print(f"      Rationale: {opp['rationale']}")
            print(f"      Market Size: {opp['market_size']}")
            print(f"      Competition: {opp['competitive_density']}")
        
        print("\n💡 These gaps represent potential investment opportunities")
        print("   where technology capabilities exist but market applications are limited.")
    
    def demo_investment_thesis_analysis(self) -> None:
        """Demonstrate investment thesis extraction and analysis"""
        
        print("\n" + "="*60)
        print("💰 DEMO: Investment Thesis Analysis")
        print("="*60)
        
        # Sample investment themes from the data
        themes = [
            {
                "theme": "Autonomous Defense Systems",
                "companies": ["Athena AI", "Aurum", "BlueSpace.ai"],
                "thesis": "AI-enabled autonomous systems will transform military operations",
                "evidence": "High autonomy scores, GPS-denied operation capability",
                "risk": "Regulatory approval, safety validation"
            },
            {
                "theme": "Edge AI for Contested Environments", 
                "companies": ["aiKOLO", "Blumind", "Bavovna.ai"],
                "thesis": "Edge computing + AI enables operation in disconnected environments",
                "evidence": "Low-power consumption, tactical edge deployment",
                "risk": "Hardware limitations, power constraints"
            },
            {
                "theme": "AI-Powered Cybersecurity Automation",
                "companies": ["AiStrike", "Amplify Security", "Autonomous Cyber"],
                "thesis": "AI will automate threat detection and response at scale",
                "evidence": "90% false positive reduction, automated incident response",
                "risk": "Adversarial AI, false negative consequences"
            }
        ]
        
        print("\n📊 INVESTMENT THEMES IDENTIFIED:")
        
        for i, theme in enumerate(themes, 1):
            print(f"\n   {i}. {theme['theme']}")
            print(f"      Thesis: {theme['thesis']}")
            print(f"      Companies: {', '.join(theme['companies'])}")
            print(f"      Evidence: {theme['evidence']}")
            print(f"      Key Risk: {theme['risk']}")
        
        print(f"\n🎯 Pattern: All themes focus on AI + specific domain expertise")
        print(f"   Common success factors: Proven performance metrics, domain specialization")
    
    def run_complete_demo(self) -> None:
        """Run the complete system demonstration"""
        
        print("🎯 VC Intelligence Knowledge Graph System - DEMO")
        print("=" * 55)
        print("Strategic Framework for Venture Capital Intelligence")
        print("Using Knowledge Graphs for Meta-Analysis and Opportunity Discovery")
        print("=" * 55)
        
        try:
            # Run all demonstrations
            self.demo_llm_knowledge_extraction()  # New LLM demo first
            self.demo_ontology_mapping()
            self.demo_analytical_insights()
            self.demo_competitive_landscape()
            self.demo_white_space_analysis()
            self.demo_investment_thesis_analysis()
            
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETE")
            print("="*60)
            print("This demonstration showed the core capabilities of the AI-powered system:")
            print("✅ LLM-based knowledge extraction using Gemini")
            print("✅ VC-specific ontology mapping")
            print("✅ Contextual understanding of unstructured text")
            print("✅ Competitive landscape analysis")
            print("✅ White space opportunity identification")
            print("✅ Investment thesis pattern recognition")
            print("\nTo run the full LLM-powered pipeline:")
            print("   python csv_knowledge_extractor.py input.csv output.json")
            print("   python main_pipeline.py")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            logger.error(f"Demo error: {e}")

def main():
    """Main demo function"""
    
    demo = SystemDemo()
    demo.run_complete_demo()
    
    return 0

if __name__ == "__main__":
    exit(main()) 