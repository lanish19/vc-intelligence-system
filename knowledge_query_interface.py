#!/usr/bin/env python3
"""
Knowledge Graph Query Interface

This module provides a sophisticated natural language interface for querying the VC 
knowledge graph. It implements advanced querying capabilities that translate business 
questions into graph queries and provide actionable insights.

Features:
- Natural language query processing
- Intelligent query translation
- Multi-modal query results
- Business intelligence interpretation
- Strategic insight generation

Author: AI Mapping Knowledge Graph System
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Container for query results with business interpretation"""
    query: str
    results: List[Dict[str, Any]]
    interpretation: str
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    execution_time: float

@dataclass
class QueryPattern:
    """Pattern for matching natural language queries"""
    pattern: str
    query_type: str
    cypher_template: str
    interpretation_template: str

class KnowledgeQueryInterface:
    """
    Natural language interface for the VC knowledge graph.
    
    Translates business questions into graph queries and provides
    strategic insights from the results.
    """
    
    def __init__(self, neo4j_driver=None):
        """Initialize with optional Neo4j connection"""
        self.driver = neo4j_driver
        self.query_patterns = self._initialize_query_patterns()
        self.query_history = []
        
    def _initialize_query_patterns(self) -> List[QueryPattern]:
        """Initialize common VC query patterns"""
        return [
            # Competitive analysis patterns
            QueryPattern(
                pattern=r"who (?:are|competes with|compete with) (.+?)(?:\s+competitors?)?",
                query_type="competitors",
                cypher_template="""
                MATCH (company:Entity {name: '{company}'})-[:RELATES_TO]->(competitor:Entity)
                WHERE competitor.type = 'Company' AND exists((company)-[:RELATES_TO {type: 'competes_with'}]->(competitor))
                RETURN competitor.name as competitor_name, 
                       count(*) as relationship_strength
                ORDER BY relationship_strength DESC
                LIMIT 10
                """,
                interpretation_template="Found {count} direct competitors for {company}"
            ),
            
            # Technology analysis patterns  
            QueryPattern(
                pattern=r"what (?:technologies|tech) (?:does|do) (.+?) (?:use|employ|leverage)",
                query_type="technologies",
                cypher_template="""
                MATCH (company:Entity {name: '{company}'})-[:RELATES_TO]->(tech:Entity)
                WHERE tech.type = 'Technology'
                RETURN tech.name as technology, 
                       collect(DISTINCT r.type) as relationship_types
                """,
                interpretation_template="{company} uses {count} key technologies"
            ),
            
            # Market analysis patterns
            QueryPattern(
                pattern=r"(?:which|what) (?:companies|startups) (?:target|focus on|operate in) (.+?)(?:\s+market)?",
                query_type="market_players",
                cypher_template="""
                MATCH (company:Entity)-[:RELATES_TO]->(market:Entity {name: '{market}'})
                WHERE company.type = 'Company' AND market.type = 'Market'
                RETURN company.name as company_name,
                       count(*) as market_focus_strength
                ORDER BY market_focus_strength DESC
                LIMIT 15
                """,
                interpretation_template="Found {count} companies targeting the {market} market"
            ),
            
            # Investment thesis patterns
            QueryPattern(
                pattern=r"what (?:is|are) (.+?)(?:'s)? (?:thesis|investment thesis|hypothesis)",
                query_type="thesis",
                cypher_template="""
                MATCH (company:Entity {name: '{company}'})-[:RELATES_TO]->(thesis:Entity)
                WHERE thesis.type = 'ThesisConcept'
                RETURN thesis.name as thesis_concept,
                       count(*) as confidence
                ORDER BY confidence DESC
                """,
                interpretation_template="Investment thesis for {company}"
            ),
            
            # Problem/friction analysis
            QueryPattern(
                pattern=r"what (?:problems?|frictions?|issues?) (?:does|do) (.+?) (?:solve|address|fix)",
                query_type="problems",
                cypher_template="""
                MATCH (company:Entity {name: '{company}'})-[:RELATES_TO]->(friction:Entity)
                WHERE friction.type = 'MarketFriction'
                RETURN friction.name as problem_addressed,
                       count(*) as focus_intensity
                ORDER BY focus_intensity DESC
                """,
                interpretation_template="{company} addresses {count} key market problems"
            ),
            
            # Differentiation analysis
            QueryPattern(
                pattern=r"(?:what|how) (?:makes|is) (.+?) (?:different|unique|special)",
                query_type="differentiators",
                cypher_template="""
                MATCH (company:Entity {name: '{company}'})-[:RELATES_TO]->(diff:Entity)
                WHERE diff.type = 'Differentiator'
                RETURN diff.name as differentiator,
                       count(*) as emphasis
                ORDER BY emphasis DESC
                """,
                interpretation_template="Key differentiators for {company}"
            ),
            
            # Sector analysis patterns
            QueryPattern(
                pattern=r"(?:which|what) (?:companies|startups) (?:are|work) in (.+?)(?:\s+(?:sector|industry|space))?",
                query_type="sector_analysis",
                cypher_template="""
                MATCH (company:Entity)-[:RELATES_TO]->(sector:Entity)
                WHERE sector.type = 'Market' AND toLower(sector.name) CONTAINS toLower('{sector}')
                RETURN company.name as company_name,
                       sector.name as sector_name,
                       count(*) as sector_focus
                ORDER BY sector_focus DESC
                LIMIT 20
                """,
                interpretation_template="Companies in the {sector} sector"
            ),
            
            # White space analysis
            QueryPattern(
                pattern=r"(?:what|where) (?:are|is) (?:the )?(?:gaps?|opportunities?|white space) in (.+?)",
                query_type="white_space",
                cypher_template="""
                MATCH (market:Entity {type: 'Market'})-[:RELATES_TO]-(company:Entity {type: 'Company'})
                WHERE toLower(market.name) CONTAINS toLower('{market}')
                WITH market, count(company) as company_count
                ORDER BY company_count ASC
                RETURN market.name as underserved_market,
                       company_count as current_players
                LIMIT 10
                """,
                interpretation_template="Potential white space opportunities in {market}"
            )
        ]
    
    def process_natural_language_query(self, query: str, 
                                     data_source: Optional[List[Dict]] = None) -> QueryResult:
        """
        Process a natural language query and return structured results.
        
        Args:
            query: Natural language question
            data_source: Optional data source (if no Neo4j available)
            
        Returns:
            QueryResult with findings and business interpretation
        """
        
        logger.info(f"Processing query: {query}")
        start_time = datetime.now()
        
        # Match query pattern
        matched_pattern = self._match_query_pattern(query)
        
        if not matched_pattern:
            return self._handle_unstructured_query(query, data_source)
        
        # Extract parameters from query
        parameters = self._extract_query_parameters(query, matched_pattern)
        
        # Execute query
        if self.driver and matched_pattern.cypher_template:
            results = self._execute_cypher_query(matched_pattern, parameters)
        else:
            results = self._execute_data_query(matched_pattern, parameters, data_source)
        
        # Generate insights and interpretation
        interpretation = self._generate_interpretation(matched_pattern, parameters, results)
        insights = self._generate_insights(matched_pattern.query_type, results, parameters)
        recommendations = self._generate_recommendations(matched_pattern.query_type, results, parameters)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        query_result = QueryResult(
            query=query,
            results=results,
            interpretation=interpretation,
            insights=insights,
            recommendations=recommendations,
            confidence_score=self._calculate_confidence(results, matched_pattern),
            execution_time=execution_time
        )
        
        self.query_history.append(query_result)
        return query_result
    
    def _match_query_pattern(self, query: str) -> Optional[QueryPattern]:
        """Match query against known patterns"""
        query_lower = query.lower().strip()
        
        for pattern in self.query_patterns:
            if re.search(pattern.pattern, query_lower, re.IGNORECASE):
                return pattern
        
        return None
    
    def _extract_query_parameters(self, query: str, pattern: QueryPattern) -> Dict[str, str]:
        """Extract parameters from query using pattern"""
        query_lower = query.lower().strip()
        match = re.search(pattern.pattern, query_lower, re.IGNORECASE)
        
        parameters = {}
        if match:
            if pattern.query_type in ['competitors', 'technologies', 'thesis', 'problems', 'differentiators']:
                parameters['company'] = match.group(1).strip()
            elif pattern.query_type in ['market_players']:
                parameters['market'] = match.group(1).strip()
            elif pattern.query_type in ['sector_analysis']:
                parameters['sector'] = match.group(1).strip()
            elif pattern.query_type in ['white_space']:
                parameters['market'] = match.group(1).strip()
        
        return parameters
    
    def _execute_data_query(self, pattern: QueryPattern, parameters: Dict[str, str], 
                          data_source: List[Dict]) -> List[Dict[str, Any]]:
        """Execute query against in-memory data when Neo4j unavailable"""
        
        if not data_source:
            return []
        
        # Convert to DataFrame for easier querying
        df = pd.DataFrame(data_source)
        results = []
        
        try:
            if pattern.query_type == 'competitors':
                company = parameters.get('company', '').lower()
                # Find companies with similar technologies or markets
                company_data = df[df['subject'].str.lower().str.contains(company, na=False)]
                if not company_data.empty:
                    # Get related entities
                    objects = company_data['object'].unique()
                    # Find other companies with same objects
                    competitors = df[
                        (df['object'].isin(objects)) & 
                        (~df['subject'].str.lower().str.contains(company, na=False)) &
                        (df['subject_type'] == 'Company')
                    ]['subject'].value_counts().head(10)
                    
                    results = [{'competitor_name': comp, 'relationship_strength': count} 
                             for comp, count in competitors.items()]
            
            elif pattern.query_type == 'technologies':
                company = parameters.get('company', '').lower()
                company_data = df[
                    (df['subject'].str.lower().str.contains(company, na=False)) &
                    (df['object_type'] == 'Technology')
                ]
                techs = company_data['object'].value_counts()
                results = [{'technology': tech, 'relationship_types': ['uses_technology']} 
                         for tech in techs.index]
            
            elif pattern.query_type == 'market_players':
                market = parameters.get('market', '').lower()
                market_data = df[
                    (df['object'].str.lower().str.contains(market, na=False)) &
                    (df['object_type'] == 'Market')
                ]
                companies = market_data['subject'].value_counts().head(15)
                results = [{'company_name': comp, 'market_focus_strength': count} 
                         for comp, count in companies.items()]
            
            elif pattern.query_type == 'sector_analysis':
                sector = parameters.get('sector', '').lower()
                sector_data = df[
                    (df['object'].str.lower().str.contains(sector, na=False)) &
                    (df['object_type'] == 'Market')
                ]
                companies = sector_data.groupby(['subject', 'object']).size().reset_index(name='sector_focus')
                companies = companies.sort_values('sector_focus', ascending=False).head(20)
                results = [
                    {
                        'company_name': row['subject'], 
                        'sector_name': row['object'],
                        'sector_focus': row['sector_focus']
                    } 
                    for _, row in companies.iterrows()
                ]
            
            else:
                # Generic search for other patterns
                param_value = list(parameters.values())[0] if parameters else ''
                if param_value:
                    filtered = df[df['subject'].str.lower().str.contains(param_value.lower(), na=False)]
                    results = filtered.head(10).to_dict('records')
        
        except Exception as e:
            logger.error(f"Error executing data query: {e}")
            results = []
        
        return results
    
    def _generate_interpretation(self, pattern: QueryPattern, parameters: Dict[str, str], 
                               results: List[Dict]) -> str:
        """Generate business interpretation of results"""
        
        try:
            template = pattern.interpretation_template
            param_dict = {**parameters, 'count': len(results)}
            return template.format(**param_dict)
        except:
            return f"Found {len(results)} results for your query"
    
    def _generate_insights(self, query_type: str, results: List[Dict], 
                         parameters: Dict[str, str]) -> List[str]:
        """Generate strategic insights from results"""
        
        insights = []
        
        if not results:
            insights.append("No direct matches found - this could indicate a white space opportunity")
            return insights
        
        if query_type == 'competitors':
            if len(results) > 7:
                insights.append("High competitive density - market may be saturated")
            elif len(results) < 3:
                insights.append("Low competitive density - potential blue ocean opportunity")
            
            top_competitor = results[0] if results else None
            if top_competitor and top_competitor.get('relationship_strength', 0) > 5:
                insights.append(f"Strong competitive overlap with {top_competitor['competitor_name']}")
        
        elif query_type == 'technologies':
            if len(results) > 5:
                insights.append("Technology-diverse company - potential platform play")
            elif len(results) <= 2:
                insights.append("Focused technology approach - specialized solution")
            
            # Look for emerging technologies
            ai_techs = [r for r in results if 'ai' in r['technology'].lower() or 'ml' in r['technology'].lower()]
            if ai_techs:
                insights.append("Strong AI/ML technology focus - aligns with current market trends")
        
        elif query_type == 'market_players':
            if len(results) > 15:
                insights.append("Crowded market with many players")
            elif len(results) < 5:
                insights.append("Emerging market with few established players")
        
        elif query_type == 'white_space':
            underserved = [r for r in results if r.get('current_players', 0) < 3]
            if underserved:
                insights.append(f"Identified {len(underserved)} potentially underserved market segments")
        
        return insights
    
    def _generate_recommendations(self, query_type: str, results: List[Dict], 
                                parameters: Dict[str, str]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if query_type == 'competitors':
            if len(results) > 5:
                recommendations.append("Conduct detailed competitive positioning analysis")
                recommendations.append("Consider differentiation strategy to avoid head-to-head competition")
            else:
                recommendations.append("Investigate market expansion opportunities")
        
        elif query_type == 'technologies':
            recommendations.append("Assess technology stack for potential IP or partnership opportunities")
            if len(results) > 3:
                recommendations.append("Evaluate technology integration risks and benefits")
        
        elif query_type == 'market_players':
            if len(results) < 5:
                recommendations.append("Consider first-mover advantage strategies")
            else:
                recommendations.append("Analyze market segmentation for niche opportunities")
        
        elif query_type == 'white_space':
            recommendations.append("Investigate customer needs in underserved segments")
            recommendations.append("Assess market size and growth potential for identified gaps")
        
        return recommendations
    
    def _calculate_confidence(self, results: List[Dict], pattern: QueryPattern) -> float:
        """Calculate confidence score for results"""
        
        if not results:
            return 0.1
        
        # Base confidence on result count and pattern specificity
        result_factor = min(len(results) / 10.0, 1.0)  # More results = higher confidence
        pattern_factor = 0.8  # Base pattern matching confidence
        
        return min(result_factor * pattern_factor, 0.95)
    
    def _handle_unstructured_query(self, query: str, data_source: List[Dict]) -> QueryResult:
        """Handle queries that don't match predefined patterns"""
        
        # Simple keyword-based search
        if not data_source:
            return QueryResult(
                query=query,
                results=[],
                interpretation="No data source available for unstructured query",
                insights=["Consider rephrasing your query to match supported patterns"],
                recommendations=["Use more specific terms like 'competitors', 'technologies', or 'market'"],
                confidence_score=0.1,
                execution_time=0.0
            )
        
        # Extract keywords and search
        keywords = [word.lower() for word in re.findall(r'\b\w+\b', query) if len(word) > 3]
        df = pd.DataFrame(data_source)
        
        results = []
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            matches = df[
                df['subject'].str.lower().str.contains(keyword, na=False) |
                df['object'].str.lower().str.contains(keyword, na=False)
            ].head(5)
            results.extend(matches.to_dict('records'))
        
        return QueryResult(
            query=query,
            results=results[:10],  # Limit results
            interpretation=f"Keyword search found {len(results)} potential matches",
            insights=["Consider using more specific query patterns for better results"],
            recommendations=["Try queries like 'Who competes with X?' or 'What technologies does Y use?'"],
            confidence_score=0.3,
            execution_time=0.1
        )
    
    def get_query_suggestions(self) -> List[str]:
        """Get suggested queries based on common VC analysis needs"""
        
        return [
            "Who competes with [company name]?",
            "What technologies does [company name] use?",
            "Which companies target the cybersecurity market?",
            "What is [company name]'s investment thesis?",
            "What problems does [company name] solve?",
            "How is [company name] different from competitors?",
            "Which companies are in the AI sector?",
            "What are the gaps in the autonomous systems market?",
            "Who are the key players in defense technology?",
            "What makes [company name] unique?"
        ]
    
    def format_results_for_display(self, result: QueryResult) -> str:
        """Format query results for human-readable display"""
        
        output = f"""
🔍 QUERY: {result.query}

📊 INTERPRETATION: {result.interpretation}

📈 RESULTS ({len(result.results)} found):
"""
        
        for i, item in enumerate(result.results[:10], 1):
            output += f"   {i}. "
            if 'competitor_name' in item:
                output += f"{item['competitor_name']} (strength: {item.get('relationship_strength', 'N/A')})\n"
            elif 'technology' in item:
                output += f"{item['technology']}\n"
            elif 'company_name' in item:
                output += f"{item['company_name']}\n"
            else:
                # Generic formatting
                key_field = next((k for k in item.keys() if 'name' in k.lower()), list(item.keys())[0])
                output += f"{item.get(key_field, 'Unknown')}\n"
        
        if result.insights:
            output += f"\n💡 KEY INSIGHTS:\n"
            for insight in result.insights:
                output += f"   • {insight}\n"
        
        if result.recommendations:
            output += f"\n🎯 RECOMMENDATIONS:\n"
            for rec in result.recommendations:
                output += f"   • {rec}\n"
        
        output += f"\n⚡ Confidence: {result.confidence_score:.1%} | Time: {result.execution_time:.2f}s\n"
        
        return output

def main():
    """Demonstration of the query interface"""
    
    print("🧠 VC Knowledge Graph Query Interface")
    print("=" * 50)
    
    # Initialize interface
    interface = KnowledgeQueryInterface()
    
    # Load sample data for demo
    try:
        from csv_knowledge_extractor import VCOntologyExtractor
        extractor = VCOntologyExtractor()
        triplets = extractor.process_csv("raw ai mapping data.csv")
        
        print(f"📊 Loaded {len(triplets)} knowledge triplets")
        
        # Demo queries
        demo_queries = [
            "Who competes with Athena AI?",
            "What technologies does aiKOLO use?",
            "Which companies target the cybersecurity market?",
            "What are the gaps in the autonomous systems market?"
        ]
        
        for query in demo_queries:
            print(f"\n" + "="*60)
            result = interface.process_natural_language_query(query, triplets)
            print(interface.format_results_for_display(result))
        
        print("\n🎯 QUERY SUGGESTIONS:")
        for suggestion in interface.get_query_suggestions():
            print(f"   • {suggestion}")
    
    except Exception as e:
        print(f"Error in demo: {e}")

if __name__ == "__main__":
    main() 