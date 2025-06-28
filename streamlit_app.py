#!/usr/bin/env python3
"""
VC Intelligence Streamlit App

A comprehensive web interface for the AI-powered VC Intelligence System.
Features:
1. Traditional pipeline functionality with UI
2. Interactive chatbot for natural language queries
3. File upload and processing
4. Real-time visualizations
5. Knowledge graph exploration

Author: AI Mapping Knowledge Graph System
"""

import streamlit as st
import pandas as pd
import json
import logging
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import traceback

# Import system components
from csv_knowledge_extractor import VCOntologyExtractor
from llm_client import KnowledgeExtractor
from graph_analytics import GraphAnalytics
from graph_visualizer import GraphVisualizer
from config import get_config, validate_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="VC Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VCIntelligenceApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.setup_session_state()
        self.load_components()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'extracted_data' not in st.session_state:
            st.session_state.extracted_data = None
        
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = "ready"
        
        if 'config_valid' not in st.session_state:
            st.session_state.config_valid = False
    
    def load_components(self):
        """Load system components"""
        try:
            # Validate configuration
            config_status = validate_config()
            st.session_state.config_valid = config_status['valid']
            
            if st.session_state.config_valid:
                self.extractor = VCOntologyExtractor()
                self.llm_extractor = KnowledgeExtractor()
                self.analytics = GraphAnalytics()
                self.visualizer = GraphVisualizer()
            else:
                st.error("⚠️ Configuration issues detected:")
                for issue in config_status['issues']:
                    st.error(f"   • {issue}")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {str(e)}")
            st.session_state.config_valid = False
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        # 🧠 VC Intelligence System
        **AI-Powered Knowledge Graph for Venture Capital Intelligence**
        
        Transform your deal flow into actionable insights using Google's Gemini AI and advanced graph analytics.
        """)
        
        # Configuration status
        if st.session_state.config_valid:
            st.success("✅ System ready - Gemini API connected")
        else:
            st.error("❌ Configuration required - Set GEMINI_API_KEY")
            with st.expander("🔧 Setup Instructions"):
                st.markdown("""
                1. Get your Gemini API key from: https://aistudio.google.com/app/apikey
                2. Set environment variable: `export GEMINI_API_KEY=your_key_here`
                3. Or create a `.env` file with: `GEMINI_API_KEY=your_key_here`
                4. Restart the Streamlit app
                """)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        st.sidebar.markdown("## 🚀 Navigation")
        
        # Mode selection
        mode = st.sidebar.radio(
            "Choose Mode:",
            ["📊 Pipeline Mode", "💬 Chat Mode", "📈 Analytics", "🔍 Exploration"],
            help="Select how you want to interact with the system"
        )
        
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.markdown("### 🔧 System Status")
        if st.session_state.config_valid:
            st.sidebar.success("AI Engine: Connected")
        else:
            st.sidebar.error("AI Engine: Not configured")
        
        # Data status
        if st.session_state.extracted_data:
            st.sidebar.success(f"Data: {len(st.session_state.extracted_data)} triplets loaded")
        else:
            st.sidebar.info("Data: No data loaded")
        
        return mode
    
    def render_pipeline_mode(self):
        """Render the traditional pipeline interface"""
        st.markdown("## 📊 Knowledge Extraction Pipeline")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with company data",
            type=['csv'],
            help="Upload your deal flow CSV for knowledge extraction"
        )
        
        if uploaded_file is not None:
            # Load and preview data
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Processing options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_rows = st.number_input(
                    "Max rows to process",
                    min_value=1,
                    max_value=len(df),
                    value=min(10, len(df)),
                    help="Limit processing for testing"
                )
            
            with col2:
                use_llm = st.toggle(
                    "Use LLM Extraction",
                    value=True,
                    help="Use Gemini AI for knowledge extraction"
                )
            
            with col3:
                if st.button("🚀 Extract Knowledge", type="primary"):
                    if st.session_state.config_valid:
                        self.process_data(df, max_rows, use_llm)
                    else:
                        st.error("Please configure GEMINI_API_KEY first")
        
        # Display results
        if st.session_state.extracted_data:
            self.display_extraction_results()
    
    def process_data(self, df: pd.DataFrame, max_rows: int, use_llm: bool):
        """Process the uploaded data"""
        st.session_state.processing_status = "processing"
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save uploaded file temporarily
            temp_path = "temp_upload.csv"
            df.head(max_rows).to_csv(temp_path, index=False)
            
            status_text.text("🧠 Extracting knowledge using AI...")
            progress_bar.progress(25)
            
            # Extract knowledge
            triplets = self.extractor.process_csv(
                temp_path,
                use_llm=use_llm,
                max_rows=max_rows
            )
            
            progress_bar.progress(75)
            status_text.text("📊 Processing results...")
            
            # Store results
            st.session_state.extracted_data = triplets
            
            progress_bar.progress(100)
            status_text.text("✅ Extraction complete!")
            
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
            
            st.success(f"🎉 Extracted {len(triplets)} knowledge triplets!")
            st.session_state.processing_status = "complete"
            
        except Exception as e:
            st.error(f"❌ Extraction failed: {str(e)}")
            st.session_state.processing_status = "error"
            logger.error(f"Processing error: {traceback.format_exc()}")
    
    def display_extraction_results(self):
        """Display the extracted knowledge results"""
        st.markdown("### 📈 Extraction Results")
        
        data = st.session_state.extracted_data
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Triplets", len(data))
        
        with col2:
            unique_subjects = len(set(t['subject'] for t in data))
            st.metric("Companies", unique_subjects)
        
        with col3:
            unique_predicates = len(set(t['predicate'] for t in data))
            st.metric("Relationship Types", unique_predicates)
        
        with col4:
            avg_confidence = sum(t.get('confidence', 0.8) for t in data) / len(data)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Visualizations
        self.render_extraction_visualizations(data)
        
        # Raw data view
        with st.expander("🔍 View Raw Data"):
            df_results = pd.DataFrame(data)
            st.dataframe(df_results, use_container_width=True)
    
    def render_extraction_visualizations(self, data: List[Dict]):
        """Render visualizations of extracted data"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Entity types distribution
            entity_types = [t['object_type'] for t in data]
            entity_counts = pd.Series(entity_types).value_counts()
            
            fig = px.pie(
                values=entity_counts.values,
                names=entity_counts.index,
                title="Entity Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Relationship types distribution
            predicates = [t['predicate'] for t in data]
            pred_counts = pd.Series(predicates).value_counts().head(10)
            
            fig = px.bar(
                x=pred_counts.values,
                y=pred_counts.index,
                orientation='h',
                title="Top Relationship Types"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_chat_mode(self):
        """Render the interactive chatbot interface"""
        st.markdown("## 💬 VC Intelligence Chatbot")
        st.markdown("Ask questions about your data and let AI do the analysis!")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"**🧑 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 AI:** {message['content']}")
                
                if 'data' in message:
                    # Display any data visualizations
                    self.render_chat_data(message['data'])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your VC data..."):
            self.handle_chat_input(prompt)
    
    def handle_chat_input(self, prompt: str):
        """Handle user chat input"""
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        # Process the query
        with st.spinner("🧠 Thinking..."):
            response = self.process_chat_query(prompt)
        
        # Add AI response to history
        st.session_state.chat_history.append(response)
        
        # Rerun to update the display
        st.rerun()
    
    def process_chat_query(self, query: str) -> Dict:
        """Process a natural language query"""
        query_lower = query.lower()
        
        # Query classification and routing
        if any(word in query_lower for word in ['extract', 'analyze', 'process']):
            return self.handle_extraction_query(query)
        
        elif any(word in query_lower for word in ['visualize', 'show', 'plot', 'chart']):
            return self.handle_visualization_query(query)
        
        elif any(word in query_lower for word in ['companies', 'firms', 'startups']):
            return self.handle_company_query(query)
        
        elif any(word in query_lower for word in ['technology', 'tech', 'ai', 'ml']):
            return self.handle_technology_query(query)
        
        elif any(word in query_lower for word in ['market', 'sector', 'industry']):
            return self.handle_market_query(query)
        
        else:
            return self.handle_general_query(query)
    
    def handle_extraction_query(self, query: str) -> Dict:
        """Handle data extraction queries"""
        if not st.session_state.config_valid:
            return {
                'role': 'assistant',
                'content': "❌ I need the GEMINI_API_KEY to be configured first. Please set up your API key in the configuration."
            }
        
        return {
            'role': 'assistant',
            'content': "🚀 I can help you extract knowledge from your data! Please upload a CSV file in the Pipeline Mode, and I'll use AI to extract insights about companies, technologies, markets, and competitive landscapes."
        }
    
    def handle_visualization_query(self, query: str) -> Dict:
        """Handle visualization queries"""
        if not st.session_state.extracted_data:
            return {
                'role': 'assistant',
                'content': "📊 I'd love to create visualizations! First, let's extract some knowledge from your data. Upload a CSV file in Pipeline Mode and I'll generate interactive charts and graphs."
            }
        
        # Create a sample visualization
        data = st.session_state.extracted_data
        entity_types = [t['object_type'] for t in data]
        entity_counts = pd.Series(entity_types).value_counts()
        
        return {
            'role': 'assistant',
            'content': f"📈 Here's a visualization of your extracted data! I found {len(data)} knowledge triplets across {len(entity_counts)} different entity types.",
            'data': {
                'type': 'pie_chart',
                'title': 'Entity Distribution',
                'data': entity_counts.to_dict()
            }
        }
    
    def handle_company_query(self, query: str) -> Dict:
        """Handle company-related queries"""
        if not st.session_state.extracted_data:
            return {
                'role': 'assistant',
                'content': "🏢 I don't have any company data loaded yet. Upload your deal flow CSV and I'll analyze all the companies for you!"
            }
        
        data = st.session_state.extracted_data
        companies = list(set(t['subject'] for t in data if t['subject_type'] == 'Company'))
        
        return {
            'role': 'assistant',
            'content': f"🏢 I found {len(companies)} companies in your data! Some examples: {', '.join(companies[:5])}{'...' if len(companies) > 5 else ''}. I can analyze their technologies, markets, differentiators, and competitive positions."
        }
    
    def handle_technology_query(self, query: str) -> Dict:
        """Handle technology-related queries"""
        if not st.session_state.extracted_data:
            return {
                'role': 'assistant',
                'content': "💻 I can analyze technology trends once you upload data! I'll identify AI/ML technologies, cybersecurity solutions, autonomous systems, and more."
            }
        
        data = st.session_state.extracted_data
        tech_triplets = [t for t in data if t['object_type'] == 'Technology']
        technologies = list(set(t['object'] for t in tech_triplets))
        
        return {
            'role': 'assistant',
            'content': f"💻 I identified {len(technologies)} different technologies! Key areas include: {', '.join(technologies[:5])}{'...' if len(technologies) > 5 else ''}. I can show you which companies use each technology and find patterns."
        }
    
    def handle_market_query(self, query: str) -> Dict:
        """Handle market-related queries"""
        if not st.session_state.extracted_data:
            return {
                'role': 'assistant',
                'content': "🎯 I can analyze market sectors and opportunities once you provide data! I'll identify target markets, market frictions, and white space opportunities."
            }
        
        data = st.session_state.extracted_data
        market_triplets = [t for t in data if t['object_type'] == 'Market']
        markets = list(set(t['object'] for t in market_triplets))
        
        return {
            'role': 'assistant',
            'content': f"🎯 I found {len(markets)} market segments! Key markets: {', '.join(markets[:3])}{'...' if len(markets) > 3 else ''}. I can analyze market concentration, identify underserved segments, and find competitive gaps."
        }
    
    def handle_general_query(self, query: str) -> Dict:
        """Handle general queries"""
        return {
            'role': 'assistant',
            'content': """🤖 I'm your VC Intelligence AI assistant! I can help you:

📊 **Extract Knowledge**: Upload CSV data and I'll use AI to extract insights
💹 **Analyze Markets**: Identify sectors, opportunities, and competitive landscapes  
🏢 **Profile Companies**: Understand technologies, differentiators, and positioning
🔍 **Find Patterns**: Discover investment themes and white space opportunities
📈 **Create Visualizations**: Generate interactive charts and graphs

What would you like to explore first?"""
        }
    
    def render_chat_data(self, data: Dict):
        """Render data visualizations in chat"""
        if data['type'] == 'pie_chart':
            fig = px.pie(
                values=list(data['data'].values()),
                names=list(data['data'].keys()),
                title=data['title']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_mode(self):
        """Render the analytics dashboard"""
        st.markdown("## 📈 Advanced Analytics")
        
        if not st.session_state.extracted_data:
            st.info("📊 No data available. Please extract knowledge first in Pipeline Mode.")
            return
        
        data = st.session_state.extracted_data
        
        # Analytics options
        analysis_type = st.selectbox(
            "Choose Analysis Type:",
            ["Overview", "Company Analysis", "Technology Trends", "Market Analysis", "Competitive Landscape"]
        )
        
        if analysis_type == "Overview":
            self.render_overview_analytics(data)
        elif analysis_type == "Company Analysis":
            self.render_company_analytics(data)
        elif analysis_type == "Technology Trends":
            self.render_technology_analytics(data)
        elif analysis_type == "Market Analysis":
            self.render_market_analytics(data)
        elif analysis_type == "Competitive Landscape":
            self.render_competitive_analytics(data)
    
    def render_overview_analytics(self, data: List[Dict]):
        """Render overview analytics"""
        st.markdown("### 📊 Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        companies = [t for t in data if t['subject_type'] == 'Company']
        technologies = [t for t in data if t['object_type'] == 'Technology']
        markets = [t for t in data if t['object_type'] == 'Market']
        differentiators = [t for t in data if t['object_type'] == 'Differentiator']
        
        with col1:
            st.metric("Companies", len(set(t['subject'] for t in companies)))
        with col2:
            st.metric("Technologies", len(set(t['object'] for t in technologies)))
        with col3:
            st.metric("Markets", len(set(t['object'] for t in markets)))
        with col4:
            st.metric("Differentiators", len(set(t['object'] for t in differentiators)))
        
        # Network summary
        st.markdown("### 🕸️ Knowledge Network Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            # Relationship types
            predicates = [t['predicate'] for t in data]
            pred_counts = pd.Series(predicates).value_counts()
            
            fig = px.bar(
                x=pred_counts.values,
                y=pred_counts.index,
                orientation='h',
                title="Relationship Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Entity types
            entity_types = [t['object_type'] for t in data]
            entity_counts = pd.Series(entity_types).value_counts()
            
            fig = px.pie(
                values=entity_counts.values,
                names=entity_counts.index,
                title="Entity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_company_analytics(self, data: List[Dict]):
        """Render company-specific analytics"""
        st.markdown("### 🏢 Company Analysis")
        
        companies = list(set(t['subject'] for t in data if t['subject_type'] == 'Company'))
        
        selected_company = st.selectbox("Select Company:", companies)
        
        if selected_company:
            company_data = [t for t in data if t['subject'] == selected_company]
            
            st.markdown(f"#### Analysis for {selected_company}")
            
            # Company metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Relationships", len(company_data))
            with col2:
                tech_count = len([t for t in company_data if t['object_type'] == 'Technology'])
                st.metric("Technologies", tech_count)
            with col3:
                market_count = len([t for t in company_data if t['object_type'] == 'Market'])
                st.metric("Markets", market_count)
            
            # Company relationships
            df_company = pd.DataFrame(company_data)
            st.dataframe(df_company[['predicate', 'object', 'object_type', 'confidence']], use_container_width=True)
    
    def render_technology_analytics(self, data: List[Dict]):
        """Render technology trend analytics"""
        st.markdown("### 💻 Technology Trends")
        
        tech_data = [t for t in data if t['object_type'] == 'Technology']
        tech_counts = pd.Series([t['object'] for t in tech_data]).value_counts()
        
        # Top technologies
        fig = px.bar(
            x=tech_counts.head(10).values,
            y=tech_counts.head(10).index,
            orientation='h',
            title="Most Common Technologies"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Technology details
        st.markdown("#### Technology Usage Details")
        tech_details = []
        for tech, count in tech_counts.head(10).items():
            companies = [t['subject'] for t in tech_data if t['object'] == tech]
            tech_details.append({
                'Technology': tech,
                'Companies': count,
                'Example Companies': ', '.join(companies[:3])
            })
        
        st.dataframe(pd.DataFrame(tech_details), use_container_width=True)
    
    def render_market_analytics(self, data: List[Dict]):
        """Render market analysis"""
        st.markdown("### 🎯 Market Analysis")
        
        market_data = [t for t in data if t['object_type'] == 'Market']
        
        if market_data:
            market_counts = pd.Series([t['object'] for t in market_data]).value_counts()
            
            # Market distribution
            fig = px.pie(
                values=market_counts.values,
                names=market_counts.index,
                title="Market Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No market data found in current dataset.")
    
    def render_competitive_analytics(self, data: List[Dict]):
        """Render competitive landscape analytics"""
        st.markdown("### ⚔️ Competitive Landscape")
        
        # Companies by technology overlap
        companies = list(set(t['subject'] for t in data if t['subject_type'] == 'Company'))
        
        if len(companies) > 1:
            st.markdown("#### Technology Overlap Analysis")
            
            # Create technology matrix
            tech_matrix = {}
            for company in companies:
                company_techs = [t['object'] for t in data 
                               if t['subject'] == company and t['object_type'] == 'Technology']
                tech_matrix[company] = set(company_techs)
            
            # Calculate overlaps
            overlap_data = []
            for i, comp1 in enumerate(companies):
                for comp2 in companies[i+1:]:
                    overlap = len(tech_matrix[comp1] & tech_matrix[comp2])
                    if overlap > 0:
                        overlap_data.append({
                            'Company 1': comp1,
                            'Company 2': comp2,
                            'Shared Technologies': overlap
                        })
            
            if overlap_data:
                st.dataframe(pd.DataFrame(overlap_data), use_container_width=True)
            else:
                st.info("No technology overlaps found between companies.")
        else:
            st.info("Need multiple companies for competitive analysis.")
    
    def render_exploration_mode(self):
        """Render the data exploration interface"""
        st.markdown("## 🔍 Data Exploration")
        
        if not st.session_state.extracted_data:
            st.info("📊 No data available. Please extract knowledge first in Pipeline Mode.")
            return
        
        data = st.session_state.extracted_data
        
        # Search and filter interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search", placeholder="Search entities...")
        
        with col2:
            entity_filter = st.selectbox(
                "Entity Type",
                ['All'] + list(set(t['object_type'] for t in data))
            )
        
        with col3:
            relation_filter = st.selectbox(
                "Relationship",
                ['All'] + list(set(t['predicate'] for t in data))
            )
        
        # Apply filters
        filtered_data = data
        
        if search_term:
            filtered_data = [t for t in filtered_data 
                           if search_term.lower() in t['subject'].lower() or 
                              search_term.lower() in t['object'].lower()]
        
        if entity_filter != 'All':
            filtered_data = [t for t in filtered_data if t['object_type'] == entity_filter]
        
        if relation_filter != 'All':
            filtered_data = [t for t in filtered_data if t['predicate'] == relation_filter]
        
        # Display filtered results
        st.markdown(f"### Results ({len(filtered_data)} triplets)")
        
        if filtered_data:
            df_filtered = pd.DataFrame(filtered_data)
            st.dataframe(df_filtered, use_container_width=True)
        else:
            st.info("No results found matching your filters.")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        if not st.session_state.config_valid:
            st.stop()
        
        # Sidebar navigation
        mode = self.render_sidebar()
        
        # Main content based on mode
        if mode == "📊 Pipeline Mode":
            self.render_pipeline_mode()
        elif mode == "💬 Chat Mode":
            self.render_chat_mode()
        elif mode == "📈 Analytics":
            self.render_analytics_mode()
        elif mode == "🔍 Exploration":
            self.render_exploration_mode()

def main():
    """Main function"""
    app = VCIntelligenceApp()
    app.run()

if __name__ == "__main__":
    main() 