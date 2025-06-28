# 🧠 VC Intelligence System - Streamlit Web App

A comprehensive web interface for the AI-powered VC Intelligence System using Google's Gemini AI and advanced graph analytics.

## ✨ Features

### 📊 Pipeline Mode
- **File Upload**: Upload CSV files with company/startup data
- **AI Extraction**: Use Gemini AI to extract knowledge triplets
- **Real-time Processing**: Watch progress with live status updates
- **Instant Results**: View extracted insights immediately

### 💬 Chat Mode
- **Natural Language Queries**: Ask questions in plain English
- **Intelligent Responses**: AI understands context and provides relevant answers
- **Data-Driven Insights**: Get answers based on your actual extracted data
- **Interactive Conversation**: Chat history maintains context

### 📈 Analytics Mode
- **Overview Dashboard**: High-level metrics and visualizations
- **Technology Trends**: Analyze tech adoption and patterns
- **Market Analysis**: Understand market segments and opportunities
- **Company Profiles**: Deep-dive into individual company data
- **Competitive Landscape**: Find overlaps and competitive insights

### 🔍 Exploration Mode
- **Advanced Search**: Filter by entity types, relationships, and keywords
- **Dynamic Filtering**: Real-time results as you adjust filters
- **Data Export**: Download filtered results as CSV
- **Detailed Views**: Explore triplets and relationships

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Gemini API Key
Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

**Option A: Environment Variable**
```bash
export GEMINI_API_KEY=your_api_key_here
```

**Option B: .env File**
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3. Launch the App
```bash
# Using the launcher (recommended)
python run_app.py

# Or directly with streamlit
streamlit run vc_app.py
```

### 4. Open in Browser
The app will automatically open at: `http://localhost:8501`

## 📋 How to Use

### Getting Started
1. **Set API Key**: Ensure your Gemini API key is configured (green checkmark in header)
2. **Upload Data**: Go to Pipeline Mode and upload a CSV file
3. **Extract Knowledge**: Click "🚀 Extract Knowledge" and watch the AI work
4. **Explore Results**: Switch between modes to analyze your data

### Sample Queries for Chat Mode
- "What companies are in my data?"
- "Show me the technologies"
- "What markets are covered?"
- "Create a visualization"
- "Tell me about competitive overlaps"

### Supported CSV Format
Your CSV should include company information such as:
- Company names
- Descriptions
- Technologies
- Market segments
- Problem statements
- Differentiators

The AI will automatically extract structured knowledge from any text fields.

## 🔧 System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB free space
- **Internet**: Required for Gemini API calls
- **Browser**: Modern browser (Chrome, Firefox, Safari, Edge)

## 🌟 Key Benefits

### 🤖 AI-Powered
- Uses Google's latest Gemini AI model
- Advanced natural language understanding
- Automatic knowledge extraction from unstructured text

### 📊 Interactive Analytics
- Real-time visualizations with Plotly
- Drill-down capabilities
- Export functionality

### 💬 Conversational Interface
- Ask questions in natural language
- Context-aware responses
- Historical chat memory

### 🔍 Advanced Exploration
- Multi-faceted filtering
- Search across all extracted entities
- Relationship discovery

## 🛠️ Troubleshooting

### API Key Issues
- Ensure `GEMINI_API_KEY` is set correctly
- Check for typos in the API key
- Verify the key has proper permissions

### Upload Issues
- Ensure CSV file is properly formatted
- Check file size (recommend < 50MB)
- Verify UTF-8 encoding

### Performance Issues
- Start with smaller datasets (< 100 rows)
- Close other browser tabs
- Refresh the page if needed

## 🎯 Example Use Cases

1. **VC Deal Flow Analysis**: Upload pitch deck data and find investment patterns
2. **Technology Trend Research**: Identify emerging tech clusters
3. **Competitive Intelligence**: Map competitor landscapes
4. **Market Opportunity Discovery**: Find white space opportunities
5. **Portfolio Analysis**: Understand portfolio company relationships

## 📞 Support

For issues or questions:
1. Check this README first
2. Verify your API key configuration
3. Try with a smaller dataset
4. Check the browser console for errors

---

**Ready to transform your VC intelligence? 🚀**

Start by setting your API key and uploading your first dataset! 