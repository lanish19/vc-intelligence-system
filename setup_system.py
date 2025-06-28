#!/usr/bin/env python3
"""
System Setup and Configuration

This script helps set up and test the VC Intelligence Knowledge Graph System.
It verifies dependencies, tests Neo4j connectivity, and prepares the environment.

Author: AI Mapping Knowledge Graph System
"""

import subprocess
import sys
import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemSetup:
    """Setup and configuration for the VC Intelligence System"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def check_python_version(self) -> bool:
        """Check if Python version is suitable"""
        version = sys.version_info
        
        if version.major != 3 or version.minor < 8:
            self.errors.append(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all required packages are installed"""
        
        required_packages = [
            'pandas', 'numpy', 'networkx', 'neo4j', 'plotly', 'matplotlib', 
            'seaborn', 'scikit-learn', 'tqdm', 'scipy', 'community'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ {package} - NOT FOUND")
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("\n💡 To install missing packages, run:")
            logger.info(f"   pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def test_neo4j_connection(self, uri: str = "bolt://localhost:7687", 
                            username: str = "neo4j", 
                            password: str = "password") -> bool:
        """Test Neo4j database connectivity"""
        
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            driver.close()
            logger.info("✅ Neo4j connection successful")
            return True
            
        except Exception as e:
            self.errors.append(f"Neo4j connection failed: {e}")
            logger.error(f"❌ Neo4j connection failed: {e}")
            return False
    
    def create_sample_config(self) -> None:
        """Create a sample configuration file"""
        
        config = {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password"
            },
            "pipeline": {
                "batch_size": 1000,
                "output_directory": "output",
                "enable_visualization": True,
                "enable_evaluation": True
            },
            "analytics": {
                "community_detection": {
                    "algorithm": "louvain",
                    "resolution": 1.0
                },
                "centrality": {
                    "calculate_all": True,
                    "top_n_nodes": 20
                },
                "link_prediction": {
                    "method": "adamic_adar",
                    "threshold": 0.1,
                    "max_predictions": 100
                }
            }
        }
        
        config_file = Path("system_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📄 Sample configuration saved to: {config_file}")
    
    def check_data_files(self) -> bool:
        """Check if required data files exist"""
        
        required_files = ["raw ai mapping data.csv"]
        missing_files = []
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                logger.error(f"❌ {file_path} - NOT FOUND")
            else:
                file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                logger.info(f"✅ {file_path} ({file_size:.1f} MB)")
        
        if missing_files:
            self.errors.append(f"Missing data files: {', '.join(missing_files)}")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies using pip"""
        
        logger.info("📦 Installing dependencies from requirements.txt...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Failed to install dependencies: {e}")
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def run_basic_test(self) -> bool:
        """Run a basic system test"""
        
        logger.info("🧪 Running basic system test...")
        
        try:
            # Test CSV extractor
            from csv_knowledge_extractor import VCOntologyExtractor
            extractor = VCOntologyExtractor()
            logger.info("✅ CSV extractor loaded")
            
            # Test analytics
            from graph_analytics import GraphAnalytics
            analytics = GraphAnalytics()
            logger.info("✅ Graph analytics loaded")
            
            # Test visualization
            from graph_visualizer import GraphVisualizer
            visualizer = GraphVisualizer()
            logger.info("✅ Visualizer loaded")
            
            logger.info("✅ Basic system test passed")
            return True
            
        except Exception as e:
            self.errors.append(f"System test failed: {e}")
            logger.error(f"❌ System test failed: {e}")
            return False
    
    def setup_neo4j_instructions(self) -> None:
        """Provide Neo4j setup instructions"""
        
        instructions = """
📋 Neo4j Setup Instructions:

1. Download Neo4j Desktop from: https://neo4j.com/download/
2. Install and create a new database
3. Set password (default username is 'neo4j')
4. Start the database
5. Install APOC plugin (optional but recommended):
   - Go to database settings
   - Add 'apoc' to plugins
   - Restart database

Alternative - Docker setup:
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

Default connection:
   - URI: bolt://localhost:7687
   - Username: neo4j
   - Password: password (change as needed)
        """
        
        print(instructions)
    
    def run_full_setup(self) -> bool:
        """Run the complete setup process"""
        
        logger.info("🚀 Starting VC Intelligence System Setup")
        logger.info("=" * 50)
        
        success = True
        
        # Check Python version
        logger.info("1️⃣ Checking Python version...")
        if not self.check_python_version():
            success = False
        
        # Install dependencies
        logger.info("\n2️⃣ Installing dependencies...")
        if Path("requirements.txt").exists():
            if not self.install_dependencies():
                success = False
        else:
            logger.warning("⚠️ requirements.txt not found, skipping dependency installation")
        
        # Check dependencies
        logger.info("\n3️⃣ Checking dependencies...")
        if not self.check_dependencies():
            success = False
        
        # Check data files
        logger.info("\n4️⃣ Checking data files...")
        if not self.check_data_files():
            success = False
        
        # Test Neo4j (optional)
        logger.info("\n5️⃣ Testing Neo4j connection...")
        if not self.test_neo4j_connection():
            self.warnings.append("Neo4j connection failed - see setup instructions")
            logger.warning("⚠️ Neo4j connection failed")
            self.setup_neo4j_instructions()
        
        # Run basic test
        logger.info("\n6️⃣ Running system test...")
        if not self.run_basic_test():
            success = False
        
        # Create config file
        logger.info("\n7️⃣ Creating configuration...")
        self.create_sample_config()
        
        # Summary
        logger.info("\n" + "=" * 50)
        
        if success and not self.errors:
            logger.info("🎉 Setup completed successfully!")
            logger.info("System is ready to run the VC Intelligence Pipeline")
            logger.info("\nNext steps:")
            logger.info("1. Ensure Neo4j is running")
            logger.info("2. Run: python main_pipeline.py")
        else:
            logger.error("❌ Setup completed with issues")
            
            if self.errors:
                logger.error("ERRORS:")
                for error in self.errors:
                    logger.error(f"  - {error}")
            
            if self.warnings:
                logger.warning("WARNINGS:")
                for warning in self.warnings:
                    logger.warning(f"  - {warning}")
        
        return success and not self.errors

def main():
    """Main setup function"""
    
    print("🔧 VC Intelligence Knowledge Graph System - Setup")
    print("=" * 55)
    
    setup = SystemSetup()
    
    # Run setup
    success = setup.run_full_setup()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 