# 🧠 VC Intelligence System

**AI-Powered Knowledge Graph for Venture Capital Intelligence**

Transform unstructured VC deal flow into actionable insights using LLM extraction and graph analytics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Google AI](https://img.shields.io/badge/Google_AI-Gemini-green.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

The VC Intelligence System revolutionizes venture capital analysis by automatically extracting structured knowledge from unstructured pitch decks, company profiles, and deal flow data. Using advanced LLM technology and graph analytics, it transforms raw data into a queryable knowledge graph that reveals investment patterns, market trends, and white space opportunities.

### ✨ Key Features

- **🤖 AI-Powered Extraction**: Uses Google's Gemini LLM to extract structured insights from unstructured text
- **📊 Interactive Web Interface**: Streamlit-based dashboard with multiple analysis modes
- **🔗 Knowledge Graph**: Neo4j integration for relationship mapping and graph analytics
- **💬 Natural Language Chat**: Ask questions about your portfolio in plain English
- **📈 Advanced Analytics**: Community detection, centrality analysis, and trend identification
- **🎯 VC-Specific Ontology**: Purpose-built for venture capital intelligence

## 🛠️ Tech Stack

- **Backend**: Python, Google Gemini AI, Neo4j
- **Frontend**: Streamlit, Plotly
- **Analytics**: NetworkX, Graph algorithms
- **Data**: CSV processing, Knowledge extraction

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Neo4j (optional, for graph storage)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lanish19/vc-intelligence-system.git
   cd vc-intelligence-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

4. **Run the application**
   ```bash
   python run_app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📱 Interface Modes

### 📊 Pipeline Mode
- Upload CSV data files
- Real-time AI extraction with progress tracking
- Instant visualization of results
- Configurable processing options

### 💬 Chat Mode
- Interactive AI chatbot
- Natural language queries about your data
- Context-aware responses
- Query classification for companies, technologies, markets

### 📈 Analytics Mode
- Overview dashboard with key metrics
- Technology trend analysis
- Market analysis with interactive charts
- Company profiles and competitive landscape

### 🔍 Exploration Mode
- Advanced search and filtering
- Multi-faceted data exploration
- CSV export functionality
- Dynamic result filtering

## 🧠 AI Capabilities

The system uses a sophisticated VC-specific ontology to extract:

- **Investment Thesis**: Core hypotheses and strategic assumptions
- **Market Friction**: Problems and pain points being addressed
- **Differentiators**: Unique value propositions and competitive moats
- **Technology Stack**: Technical capabilities and innovations
- **Target Markets**: Customer segments and go-to-market strategies
- **Competitive Landscape**: Direct and indirect competitors

## 📊 Analytics Features

- **Community Detection**: Algorithmic discovery of thematic clusters
- **Centrality Analysis**: Identification of keystone technologies and influential players
- **Trend Analysis**: Market momentum and technology adoption patterns
- **White Space Discovery**: Link prediction for opportunity identification
- **Risk Assessment**: Portfolio concentration and dependency analysis

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687  # Optional
NEO4J_USER=neo4j                 # Optional
NEO4J_PASSWORD=your_password     # Optional
```

### Customization
- Modify `vc_prompts.py` to adjust extraction prompts
- Edit `config.py` for system configuration
- Customize analytics in `graph_analytics.py`

## 📁 Project Structure

```
vc-intelligence-system/
├── 🚀 run_app.py              # Application launcher
├── 🎛️ vc_app.py               # Main Streamlit interface  
├── 🧠 llm_client.py           # Gemini LLM integration
├── 📊 csv_knowledge_extractor.py # Data processing engine
├── 🔗 graph_analytics.py      # Graph analysis algorithms
├── 📈 graph_visualizer.py     # Visualization components
├── ⚙️ config.py               # System configuration
├── 📝 vc_prompts.py           # LLM prompt engineering
├── 🔍 knowledge_query_interface.py # Query processing
├── 📋 requirements.txt        # Dependencies
└── 📖 README.md              # This file
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google AI team for Gemini LLM capabilities
- Streamlit team for the excellent web framework
- Neo4j for graph database technology
- The open-source community for foundational tools

## 📞 Contact

- **GitHub**: [@lanish19](https://github.com/lanish19)
- **Repository**: [vc-intelligence-system](https://github.com/lanish19/vc-intelligence-system)

---

**⭐ If you find this project useful, please give it a star!**

Built with ❤️ for the venture capital community 