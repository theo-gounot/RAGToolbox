from abc import ABC, abstractmethod
from typing import Any, Dict
from src.land_rag.config import RAGConfig

class AbstractBaseModule(ABC):
    """
    Abstract base class for all RAG pipeline modules.
    Enforces a consistent interface for the modular architecture.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config

    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """
        Execute the module's logic.
        
        Args:
            input_data: The input data for the module (type depends on module).
            **kwargs: Additional arguments.
            
        Returns:
            The processed output.
        """
        pass
