#!/usr/bin/env python3
"""
Hybrid LLM Client for WellWise AI

This module provides a unified interface for LLM-based metadata enhancement
that automatically switches between OpenAI (production) and Ollama (development)
based on the ENVIRONMENT variable.

Usage:
    from utils.llm_client import LLMClient
    
    client = LLMClient()
    result = client.enhance_metadata(prompt)
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Hybrid LLM client that automatically switches between OpenAI and Ollama
    based on the ENVIRONMENT variable.
    
    Environment Variables:
        ENVIRONMENT: 'production' uses OpenAI, anything else uses Ollama
        CHAT_COMPLETION_MODEL: OpenAI model (default: gpt-4o-mini)
        LOCAL_CHAT_COMPLETION_MODEL: Ollama model (default: llama3.1:8b)
    """
    
    def __init__(self):
        """Initialize the LLM client with appropriate backend."""
        self.environment = os.getenv('ENVIRONMENT', 'dev').lower()
        self.openai_model = os.getenv('CHAT_COMPLETION_MODEL', 'gpt-4o-mini')
        self.ollama_model = os.getenv('LOCAL_CHAT_COMPLETION_MODEL', 'llama3.1:8b')
        
        # Initialize backends
        self.openai_client = None
        self.ollama_available = False
        
        if self.environment == 'production':
            self._init_openai()
        else:
            self._init_ollama()
        
        logger.info(f"LLM Client initialized with {self._get_backend_name()}")
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client for production use."""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI()
            logger.info(f"OpenAI client initialized with model: {self.openai_model}")
        except ImportError:
            logger.error("OpenAI library not available")
            raise RuntimeError("OpenAI library required for production environment")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"OpenAI initialization failed: {e}")
    
    def _init_ollama(self) -> None:
        """Initialize Ollama client for development use."""
        try:
            import ollama
            self.ollama_available = True
            logger.info(f"Ollama client initialized with model: {self.ollama_model}")
        except ImportError:
            logger.warning("Ollama library not available, falling back to OpenAI")
            self._init_openai()
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}, falling back to OpenAI")
            self._init_openai()
    
    def _get_backend_name(self) -> str:
        """Get the name of the current backend."""
        if self.environment == 'production':
            return f"OpenAI ({self.openai_model})"
        else:
            return f"Ollama ({self.ollama_model})"
    
    def enhance_metadata(self, prompt: str, max_tokens: int = 800) -> Dict[str, Any]:
        """
        Enhance metadata using the appropriate LLM backend.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens for the response (default: 800)
            
        Returns:
            Dictionary containing the parsed JSON response
            
        Raises:
            RuntimeError: If both backends fail
            ValueError: If the response cannot be parsed as JSON
        """
        if self.environment == 'production':
            return self._openai_enhance(prompt, max_tokens)
        else:
            return self._ollama_enhance(prompt, max_tokens)
    
    def _openai_enhance(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Enhance metadata using OpenAI."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            logger.debug(f"Calling OpenAI with {max_tokens} max tokens")
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            logger.debug(f"OpenAI response length: {len(content)} chars")
            
            # Parse JSON response
            result = json.loads(content)
            logger.debug(f"OpenAI enhancement successful")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"OpenAI JSON parsing failed: {e}")
            logger.debug(f"Raw OpenAI response: {content}")
            raise ValueError(f"Invalid JSON response from OpenAI: {e}")
        except Exception as e:
            logger.error(f"OpenAI enhancement failed: {e}")
            raise RuntimeError(f"OpenAI call failed: {e}")
    
    def _ollama_enhance(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Enhance metadata using Ollama."""
        if not self.ollama_available:
            # Fallback to OpenAI if Ollama is not available
            logger.warning("Ollama not available, falling back to OpenAI")
            return self._openai_enhance(prompt, max_tokens)
        
        try:
            import ollama
            
            logger.debug(f"Calling Ollama with {max_tokens} max tokens")
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": "Return ONLY a single JSON object. No prose."},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={
                    "temperature": 0,
                    "num_ctx": 8192,
                    "seed": 7
                }
            )
            
            content = response["message"]["content"]
            logger.debug(f"Ollama response length: {len(content)} chars")
            
            # Parse JSON response
            result = json.loads(content)
            logger.debug(f"Ollama enhancement successful")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Ollama JSON parsing failed: {e}")
            logger.debug(f"Raw Ollama response: {content}")
            raise ValueError(f"Invalid JSON response from Ollama: {e}")
        except Exception as e:
            logger.error(f"Ollama enhancement failed: {e}")
            # Fallback to OpenAI if Ollama fails
            logger.warning("Ollama failed, falling back to OpenAI")
            try:
                return self._openai_enhance(prompt, max_tokens)
            except Exception as fallback_error:
                logger.error(f"OpenAI fallback also failed: {fallback_error}")
                raise RuntimeError(f"Both Ollama and OpenAI failed: {e}, {fallback_error}")
    
    def is_available(self) -> bool:
        """
        Check if the LLM client is available and ready to use.
        
        Returns:
            True if the client is ready, False otherwise
        """
        if self.environment == 'production':
            return self.openai_client is not None
        else:
            return self.ollama_available or self.openai_client is not None
    
    def get_backend_info(self) -> Dict[str, str]:
        """
        Get information about the current backend configuration.
        
        Returns:
            Dictionary with backend information
        """
        return {
            "environment": self.environment,
            "backend": self._get_backend_name(),
            "available": self.is_available(),
            "openai_model": self.openai_model,
            "ollama_model": self.ollama_model
        }


def create_llm_client() -> LLMClient:
    """
    Factory function to create an LLM client instance.
    
    Returns:
        Configured LLMClient instance
    """
    return LLMClient()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing LLM Client")
    print("=" * 50)
    
    try:
        # Create client
        client = create_llm_client()
        
        # Show backend info
        info = client.get_backend_info()
        print(f"Backend: {info['backend']}")
        print(f"Environment: {info['environment']}")
        print(f"Available: {info['available']}")
        
        if not client.is_available():
            print("❌ LLM client not available")
            exit(1)
        
        # Test with simple prompt
        test_prompt = """
        Extract key information and return JSON:
        
        Content: Well 15/9-F-1 Petrophysical Analysis Report
        
        Return JSON with: {"well_id": "string", "document_type": "string"}
        """
        
        print(f"\n🧪 Testing with simple prompt...")
        result = client.enhance_metadata(test_prompt)
        
        print(f"✅ Success!")
        print(f"📊 Result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
