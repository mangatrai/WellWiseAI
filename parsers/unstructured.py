"""
Unstructured Parser
Handles general document types using Unstructured SDK
Supports PDF, Excel, Word, PowerPoint, Images, and more
"""

import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from .base_parser import BaseParser, ParsedData, ContextualDocument

class UnstructuredParser(BaseParser):
    """Universal parser using Unstructured SDK for general document types"""
    
    def __init__(self, file_path: str, supported_extensions: List[str] = None):
        super().__init__(file_path)
        # Supported file extensions from configuration
        self.supported_extensions = supported_extensions or []
    
    def can_parse(self) -> bool:
        """Check if this file can be parsed by Unstructured SDK"""
        if not self.validate_file():
            return False
            
        # Skip macOS system files
        if self.file_path.name == '.DS_Store':
            return False
            
        # Check file extension if extensions are configured
        extension = self.file_path.suffix.lower()
        if self.supported_extensions:
            if extension and extension not in self.supported_extensions:
                return False
        # If no extensions configured, accept all files (safety fallback)
            
        # Try to import and test Unstructured SDK
        try:
            from unstructured.partition.auto import partition
            return True
        except ImportError:
            self.logger.error("Unstructured SDK not available")
            return False
    
    def parse(self) -> ParsedData:
        """Parse document using Unstructured SDK"""
        start_time = time.time()
        
        try:
            if not self.can_parse():
                return self.create_error_result("File cannot be parsed by Unstructured parser")
            
            # Import Unstructured SDK components
            from unstructured.partition.auto import partition
            from unstructured.chunking.title import chunk_by_title
            from unstructured.documents.elements import Table, Text, Title, Image
            
            # Process document with Unstructured
            self.logger.info(f"Processing {self.file_path.name} with Unstructured SDK")
            
            # Use text partitioner for ASC files, auto partitioner for others
            if self.file_path.suffix.lower() in ['.asc', '.txt', '.dat', '.las']:
                from unstructured.partition.text import partition_text
                elements = partition_text(str(self.file_path))
            else:
                elements = partition(str(self.file_path))
            
            # Chunk by semantic sections
            chunks = chunk_by_title(elements)
            
            # Process chunks and extract data
            data_sections = []
            element_types = {}
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': i,
                    'content': str(chunk),
                    'element_type': type(chunk).__name__,
                    'metadata': getattr(chunk, 'metadata', {}),
                    'chunk_type': 'semantic_section'
                }
                
                # Track element types for statistics
                element_type = type(chunk).__name__
                element_types[element_type] = element_types.get(element_type, 0) + 1
                
                # Add specific processing for different element types
                if isinstance(chunk, Table):
                    chunk_data['table_data'] = self._extract_table_data(chunk)
                elif isinstance(chunk, Image):
                    chunk_data['image_info'] = self._extract_image_info(chunk)
                elif isinstance(chunk, Title):
                    chunk_data['title_level'] = getattr(chunk, 'metadata', {}).get('level', 1)
                
                data_sections.append(chunk_data)
            
            # Create file structure analysis
            file_structure = {
                'total_chunks': len(chunks),
                'element_types': element_types,
                'file_type': self._detect_file_type(),
                'processing_method': 'unstructured_sdk'
            }
            
            processing_time = time.time() - start_time
            
            return ParsedData(
                file_path=str(self.file_path),
                file_type='unstructured',
                parser_name=self.__class__.__name__,
                metadata=self.get_file_info(),
                data={
                    'chunks': data_sections,
                    'file_structure': file_structure,
                    'raw_elements_count': len(elements)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing {self.file_path}: {e}")
            return self.create_error_result(str(e))
    
    def _extract_table_data(self, table_element) -> Dict[str, Any]:
        """Extract structured data from table elements"""
        try:
            # Convert table to structured format
            table_data = {
                'rows': [],
                'headers': [],
                'row_count': 0,
                'column_count': 0
            }
            
            # Extract table content (this is a simplified version)
            # In practice, you'd use table_element.metadata or table_element.text
            table_text = str(table_element)
            
            # Parse table structure
            lines = table_text.split('\n')
            if lines:
                # Assume first line contains headers
                headers = [h.strip() for h in lines[0].split('\t') if h.strip()]
                table_data['headers'] = headers
                table_data['column_count'] = len(headers)
                
                # Parse data rows
                for line in lines[1:]:
                    if line.strip():
                        row = [cell.strip() for cell in line.split('\t')]
                        table_data['rows'].append(row)
                
                table_data['row_count'] = len(table_data['rows'])
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting table data: {e}")
            return {'error': str(e)}
    
    def _extract_image_info(self, image_element) -> Dict[str, Any]:
        """Extract information from image elements"""
        try:
            image_info = {
                'image_type': 'image',
                'metadata': getattr(image_element, 'metadata', {})
            }
            
            # Extract image metadata if available
            if hasattr(image_element, 'metadata'):
                metadata = image_element.metadata
                image_info.update({
                    'width': metadata.get('width'),
                    'height': metadata.get('height'),
                    'format': metadata.get('format'),
                    'page_number': metadata.get('page_number')
                })
            
            return image_info
            
        except Exception as e:
            self.logger.warning(f"Error extracting image info: {e}")
            return {'error': str(e)}
    
    def _detect_file_type(self) -> str:
        """Detect the specific file type for better categorization"""
        extension = self.file_path.suffix.lower()
        
        file_type_mapping = {
            '.pdf': 'pdf_document',
            '.docx': 'word_document',
            '.doc': 'word_document',
            '.pptx': 'powerpoint_presentation',
            '.ppt': 'powerpoint_presentation',
            '.xlsx': 'excel_spreadsheet',
            '.xls': 'excel_spreadsheet',
            '.csv': 'csv_data',
            '.txt': 'text_document',
            '.html': 'web_document',
            '.htm': 'web_document',
            '.md': 'markdown_document',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.tiff': 'image',
            '.bmp': 'image',
            '.gif': 'image',
            '.eml': 'email',
            '.msg': 'email'
        }
        
        return file_type_mapping.get(extension, 'unknown_document')
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.supported_extensions)
        
    def generate_contextual_documents(self) -> List[ContextualDocument]:
        """Generate contextual documents from Unstructured SDK output"""
        parsed_data = self.parse()
        if parsed_data.error:
            return []
            
        documents = []
        data = parsed_data.data
        
        if not data or 'chunks' not in data:
            return []
            
        chunks = data['chunks']
        file_structure = data.get('file_structure', {})
        
        for i, chunk in enumerate(chunks):
            # Extract chunk information
            content = chunk.get('content', '')
            element_type = chunk.get('element_type', 'Unknown')
            metadata = chunk.get('metadata', {})
            
            # Create contextual document for each chunk
            # Handle metadata safely (ElementMetadata objects need to be converted to dict)
            safe_metadata = {}
            if isinstance(metadata, dict):
                safe_metadata = metadata
            else:
                # Convert ElementMetadata object to dict if needed
                try:
                    safe_metadata = dict(metadata) if hasattr(metadata, '__dict__') else {}
                except:
                    safe_metadata = {}
            
            document = ContextualDocument(
                content=content,
                metadata={
                    **parsed_data.metadata,
                    'chunk_id': i,
                    'element_type': element_type,
                    'chunk_type': chunk.get('chunk_type', 'semantic_section'),
                    'file_type': file_structure.get('file_type', 'unknown_document'),
                    'total_chunks': len(chunks),
                    'element_types': file_structure.get('element_types', {}),
                    **safe_metadata  # Include chunk-specific metadata safely
                },
                document_type=f"unstructured_{element_type.lower()}",
                source=parsed_data.file_path,
                timestamp=datetime.now().isoformat()
            )
            documents.append(document)
            
        return documents 