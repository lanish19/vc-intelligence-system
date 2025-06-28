#!/usr/bin/env python3
"""
VC Intelligence Streamlit App

A comprehensive web interface for the AI-powered VC Intelligence System.
"""

import streamlit as st

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="VC Intelligence System",
    page_icon="🧠",
    layout="wide"
)

import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import logging
import traceback
from typing import Dict, List, Any

# Import our system components
try:
    from csv_knowledge_extractor import VCOntologyExtractor
    from llm_client import KnowledgeExtractor
    from graph_analytics import GraphAnalytics
    from graph_visualizer import GraphVisualizer
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ System components not available: {e}")
    SYSTEM_AVAILABLE = False

# Header
st.title("🧠 VC Intelligence System")
st.markdown("**AI-Powered Knowledge Graph for Venture Capital Intelligence**")

# Check API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    st.success("✅ Gemini API Key configured")
else:
    st.error("❌ GEMINI_API_KEY not found. Please set your API key.")
    st.info("Set environment variable: `export GEMINI_API_KEY=your_key_here`")

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["📊 Pipeline Mode", "💬 Chat Mode", "📈 Analytics", "🔍 Exploration"]
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

def display_extraction_results(data: List[Dict]):
    """Display extracted knowledge results with visualizations"""
    if not data:
        st.info("No data to display")
        return
    
    # Key metrics
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
    col1, col2 = st.columns(2)
    
    with col1:
        # Entity types distribution
        entity_types = [t.get('object_type', 'Unknown') for t in data]
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
    
    # Raw data view
    with st.expander("🔍 View Raw Knowledge Triplets"):
        df_results = pd.DataFrame(data)
        st.dataframe(df_results, use_container_width=True)

def process_chat_query(query: str) -> str:
    """Process a natural language query and return intelligent response"""
    query_lower = query.lower()
    
    # Check if we have data
    if not st.session_state.extracted_data:
        return "🤖 I don't have any data loaded yet. Please upload a CSV file in Pipeline Mode and extract knowledge first!"
    
    data = st.session_state.extracted_data
    
    # Query classification and routing
    if any(word in query_lower for word in ['companies', 'firms', 'startups']):
        companies = list(set(t['subject'] for t in data if t.get('subject_type') == 'Company'))
        return f"🏢 I found {len(companies)} companies in your data: {', '.join(companies[:5])}{'...' if len(companies) > 5 else ''}. I can analyze their technologies, markets, and competitive positions!"
    
    elif any(word in query_lower for word in ['technology', 'tech', 'ai', 'ml']):
        tech_triplets = [t for t in data if t.get('object_type') == 'Technology']
        technologies = list(set(t['object'] for t in tech_triplets))
        return f"💻 I identified {len(technologies)} different technologies! Key areas include: {', '.join(technologies[:5])}{'...' if len(technologies) > 5 else ''}."
    
    elif any(word in query_lower for word in ['market', 'sector', 'industry']):
        market_triplets = [t for t in data if t.get('object_type') == 'Market']
        markets = list(set(t['object'] for t in market_triplets))
        return f"🎯 I found {len(markets)} market segments! Key markets: {', '.join(markets[:3])}{'...' if len(markets) > 3 else ''}."
    
    elif any(word in query_lower for word in ['visualize', 'show', 'plot', 'chart']):
        return f"📈 I can create visualizations! Check out the Analytics mode for detailed charts. I found {len(data)} knowledge triplets across multiple entity types."
    
    elif any(word in query_lower for word in ['help', 'what', 'how']):
        return """🤖 I'm your VC Intelligence AI assistant! I can help you:

📊 **Extract Knowledge**: Upload CSV data and I'll use AI to extract insights
💹 **Analyze Markets**: Identify sectors, opportunities, and competitive landscapes  
🏢 **Profile Companies**: Understand technologies, differentiators, and positioning
🔍 **Find Patterns**: Discover investment themes and white space opportunities
📈 **Create Visualizations**: Generate interactive charts and graphs

Try asking me about companies, technologies, or markets!"""
    
    else:
        # General response with data stats
        companies = len(set(t['subject'] for t in data if t.get('subject_type') == 'Company'))
        techs = len(set(t['object'] for t in data if t.get('object_type') == 'Technology'))
        return f"🤖 I understand you're asking about: '{query}'. I have {len(data)} knowledge triplets covering {companies} companies and {techs} technologies. Try asking me about specific companies, technologies, or markets!"

# Mode rendering
if mode == "📊 Pipeline Mode":
    st.header("📊 Knowledge Extraction Pipeline")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            max_rows = st.number_input("Max rows", 1, len(df), min(10, len(df)))
        with col2:
            use_llm = st.toggle("Use LLM Extraction", True)
        
        if st.button("🚀 Extract Knowledge"):
            if api_key and SYSTEM_AVAILABLE:
                try:
                    # Save uploaded file temporarily
                    temp_path = "temp_upload.csv"
                    df.head(max_rows).to_csv(temp_path, index=False)
                    
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🧠 Initializing AI extraction...")
                    progress_bar.progress(10)
                    
                    # Initialize extractor
                    extractor = VCOntologyExtractor()
                    
                    status_text.text("🔍 Extracting knowledge with Gemini AI...")
                    progress_bar.progress(25)
                    
                    # Extract knowledge
                    triplets = extractor.process_csv(
                        temp_path,
                        use_llm=use_llm,
                        max_rows=max_rows
                    )
                    
                    progress_bar.progress(75)
                    status_text.text("📊 Processing results...")
                    
                    # Store results in session state
                    st.session_state.extracted_data = triplets
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Extraction complete!")
                    
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)
                    
                    st.success(f"🎉 Successfully extracted {len(triplets)} knowledge triplets!")
                    
                    # Show results immediately
                    st.markdown("### 📈 Extraction Results")
                    display_extraction_results(triplets)
                    
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)}")
                    if Path("temp_upload.csv").exists():
                        Path("temp_upload.csv").unlink()
                    
            elif not api_key:
                st.error("Please configure GEMINI_API_KEY")
            else:
                st.error("System components not available")

elif mode == "💬 Chat Mode":
    st.header("💬 VC Intelligence Chatbot")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your VC data..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        # Generate intelligent response
        response = process_chat_query(prompt)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        
        st.rerun()

elif mode == "📈 Analytics":
    st.header("📈 Advanced Analytics")
    
    if not st.session_state.extracted_data:
        st.info("📊 Upload data in Pipeline Mode first to see analytics.")
    else:
        data = st.session_state.extracted_data
        
        # Analytics selector
        analysis_type = st.selectbox(
            "Choose Analysis Type:",
            ["Overview", "Technology Trends", "Market Analysis", "Company Profiles", "Competitive Landscape"]
        )
        
        if analysis_type == "Overview":
            st.markdown("### 📊 Data Overview")
            display_extraction_results(data)
            
        elif analysis_type == "Technology Trends":
            st.markdown("### 💻 Technology Trends")
            tech_data = [t for t in data if t.get('object_type') == 'Technology']
            
            if tech_data:
                tech_counts = pd.Series([t['object'] for t in tech_data]).value_counts()
                
                # Top technologies chart
                fig = px.bar(
                    x=tech_counts.head(10).values,
                    y=tech_counts.head(10).index,
                    orientation='h',
                    title="Most Common Technologies",
                    labels={'x': 'Number of Companies', 'y': 'Technology'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Technology details table
                st.markdown("#### Technology Details")
                tech_details = []
                for tech, count in tech_counts.head(10).items():
                    companies = [t['subject'] for t in tech_data if t['object'] == tech]
                    tech_details.append({
                        'Technology': tech,
                        'Companies Using': count,
                        'Example Companies': ', '.join(companies[:3])
                    })
                st.dataframe(pd.DataFrame(tech_details), use_container_width=True)
            else:
                st.info("No technology data found.")
                
        elif analysis_type == "Market Analysis":
            st.markdown("### 🎯 Market Analysis")
            market_data = [t for t in data if t.get('object_type') == 'Market']
            
            if market_data:
                market_counts = pd.Series([t['object'] for t in market_data]).value_counts()
                
                # Market distribution pie chart
                fig = px.pie(
                    values=market_counts.values,
                    names=market_counts.index,
                    title="Market Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Market details
                st.markdown("#### Market Segments")
                for market, count in market_counts.items():
                    companies = [t['subject'] for t in market_data if t['object'] == market]
                    st.write(f"**{market}**: {count} companies - {', '.join(companies[:3])}")
            else:
                st.info("No market data found.")
                
        elif analysis_type == "Company Profiles":
            st.markdown("### 🏢 Company Profiles")
            companies = list(set(t['subject'] for t in data if t.get('subject_type') == 'Company'))
            
            if companies:
                selected_company = st.selectbox("Select Company:", companies)
                
                if selected_company:
                    company_data = [t for t in data if t['subject'] == selected_company]
                    
                    st.markdown(f"#### Profile: {selected_company}")
                    
                    # Company metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Relationships", len(company_data))
                    with col2:
                        tech_count = len([t for t in company_data if t.get('object_type') == 'Technology'])
                        st.metric("Technologies", tech_count)
                    with col3:
                        market_count = len([t for t in company_data if t.get('object_type') == 'Market'])
                        st.metric("Markets", market_count)
                    
                    # Company details table
                    df_company = pd.DataFrame(company_data)
                    st.dataframe(df_company[['predicate', 'object', 'object_type']], use_container_width=True)
            else:
                st.info("No company data found.")
                
        elif analysis_type == "Competitive Landscape":
            st.markdown("### ⚔️ Competitive Landscape")
            companies = list(set(t['subject'] for t in data if t.get('subject_type') == 'Company'))
            
            if len(companies) > 1:
                st.markdown("#### Technology Overlap Analysis")
                
                # Create technology overlap matrix
                tech_matrix = {}
                for company in companies:
                    company_techs = [t['object'] for t in data 
                                   if t['subject'] == company and t.get('object_type') == 'Technology']
                    tech_matrix[company] = set(company_techs)
                
                # Calculate overlaps
                overlap_data = []
                for i, comp1 in enumerate(companies):
                    for comp2 in companies[i+1:]:
                        overlap = len(tech_matrix[comp1] & tech_matrix[comp2])
                        if overlap > 0:
                            shared_techs = tech_matrix[comp1] & tech_matrix[comp2]
                            overlap_data.append({
                                'Company 1': comp1,
                                'Company 2': comp2,
                                'Shared Technologies': overlap,
                                'Common Tech': ', '.join(list(shared_techs)[:3])
                            })
                
                if overlap_data:
                    st.dataframe(pd.DataFrame(overlap_data), use_container_width=True)
                else:
                    st.info("No technology overlaps found between companies.")
            else:
                st.info("Need multiple companies for competitive analysis.")

elif mode == "🔍 Exploration":
    st.header("🔍 Data Exploration")
    
    if not st.session_state.extracted_data:
        st.info("📊 Upload data in Pipeline Mode first to explore.")
    else:
        data = st.session_state.extracted_data
        
        # Search and filter interface
        st.markdown("### 🔍 Search & Filter")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("🔍 Search entities", placeholder="Enter company, tech, market...")
        
        with col2:
            entity_types = ['All'] + list(set(t.get('object_type', 'Unknown') for t in data))
            entity_filter = st.selectbox("Entity Type", entity_types)
        
        with col3:
            relationship_types = ['All'] + list(set(t['predicate'] for t in data))
            relation_filter = st.selectbox("Relationship Type", relationship_types)
        
        # Apply filters
        filtered_data = data
        
        if search_term:
            filtered_data = [t for t in filtered_data 
                           if search_term.lower() in t['subject'].lower() or 
                              search_term.lower() in t['object'].lower()]
        
        if entity_filter != 'All':
            filtered_data = [t for t in filtered_data if t.get('object_type') == entity_filter]
        
        if relation_filter != 'All':
            filtered_data = [t for t in filtered_data if t['predicate'] == relation_filter]
        
        # Display results
        st.markdown(f"### 📊 Results ({len(filtered_data)} triplets)")
        
        if filtered_data:
            # Quick stats for filtered data
            col1, col2, col3 = st.columns(3)
            with col1:
                unique_subjects = len(set(t['subject'] for t in filtered_data))
                st.metric("Unique Subjects", unique_subjects)
            with col2:
                unique_objects = len(set(t['object'] for t in filtered_data))
                st.metric("Unique Objects", unique_objects)
            with col3:
                unique_relations = len(set(t['predicate'] for t in filtered_data))
                st.metric("Relation Types", unique_relations)
            
            # Results table
            df_filtered = pd.DataFrame(filtered_data)
            
            # Display specific columns based on available data
            display_columns = ['subject', 'predicate', 'object']
            if 'object_type' in df_filtered.columns:
                display_columns.append('object_type')
            if 'confidence' in df_filtered.columns:
                display_columns.append('confidence')
            
            st.dataframe(df_filtered[display_columns], use_container_width=True)
            
            # Download option
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered results as CSV",
                data=csv,
                file_name="vc_intelligence_filtered_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No results found matching your filters. Try adjusting your search criteria.")

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 System Status")
if api_key:
    st.sidebar.success("AI Engine: Connected")
else:
    st.sidebar.error("AI Engine: Not configured")

if st.session_state.extracted_data:
    st.sidebar.success(f"Data: {len(st.session_state.extracted_data)} triplets")
else:
    st.sidebar.info("Data: No data loaded") 