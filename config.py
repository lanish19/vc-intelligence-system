#!/usr/bin/env python3
"""
Configuration Management for AI-Powered VC Intelligence System

This module handles configuration for LLM integration, API keys, and system settings.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: str = "google"
    model: str = "gemini-2.5-flash-lite-preview-06-17"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    thinking_budget: int = 0
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found! Please set it in your environment or .env file"
            )

@dataclass
class DatabaseConfig:
    """Configuration for Neo4j database"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

@dataclass
class ExtractionConfig:
    """Configuration for knowledge extraction"""
    batch_size: int = 10
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_caching: bool = True
    output_format: str = "json"

@dataclass
class SystemConfig:
    """Main system configuration"""
    llm: LLMConfig
    database: DatabaseConfig
    extraction: ExtractionConfig
    
    @classmethod
    def from_file(cls, config_path: str = "config.yaml") -> "SystemConfig":
        """Load configuration from YAML file"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls(
                llm=LLMConfig(**config_data.get('llm', {})),
                database=DatabaseConfig(**config_data.get('database', {})),
                extraction=ExtractionConfig(**config_data.get('extraction', {}))
            )
        else:
            # Return default configuration
            return cls(
                llm=LLMConfig(),
                database=DatabaseConfig(),
                extraction=ExtractionConfig()
            )
    
    def save_to_file(self, config_path: str = "config.yaml"):
        """Save configuration to YAML file"""
        config_data = {
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'thinking_budget': self.llm.thinking_budget
                # Note: API key is loaded from environment, not saved to file
            },
            'database': {
                'uri': self.database.uri,
                'user': self.database.user,
                'password': self.database.password,
                'database': self.database.database
            },
            'extraction': {
                'batch_size': self.extraction.batch_size,
                'max_retries': self.extraction.max_retries,
                'timeout_seconds': self.extraction.timeout_seconds,
                'enable_caching': self.extraction.enable_caching,
                'output_format': self.extraction.output_format
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

# Global configuration instance
config = SystemConfig.from_file()

def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config

def update_api_key(api_key: str):
    """Update the API key in the global configuration"""
    global config
    config.llm.api_key = api_key

def validate_config() -> Dict[str, Any]:
    """Validate the current configuration"""
    issues = []
    
    # Check API key
    if not config.llm.api_key:
        issues.append("GEMINI_API_KEY not set")
    
    # Check model name
    if not config.llm.model:
        issues.append("LLM model not specified")
    
    # Check database connection
    if not all([config.database.uri, config.database.user]):
        issues.append("Database configuration incomplete")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "config": config
    }

if __name__ == "__main__":
    # Test configuration
    validation = validate_config()
    print(f"Configuration valid: {validation['valid']}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    
    # Create sample config file
    config.save_to_file("config.yaml")
    print("Sample config.yaml file created") 