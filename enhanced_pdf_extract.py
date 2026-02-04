import pdfplumber
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPDFExtractor:
    """
    Enhanced PDF extractor that supports multiple PDFs with improved table and text handling.
    """
    
    def __init__(self, output_dir: str = "extracted_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.extraction_metadata = []
        
    def extract_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and tables from a single PDF with enhanced metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_metadata = {
                    'pdf_name': pdf_path.name,
                    'total_pages': len(pdf.pages),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'pages': []
                }
                
                all_text = ""
                all_tables = []
                table_counter = 0
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}/{len(pdf.pages)}")
                    
                    page_data = {
                        'page_number': page_num,
                        'text': "",
                        'tables': [],
                        'table_count': 0
                    }
                    
                    # Extract text from page
                    page_text = page.extract_text()
                    if page_text:
                        page_data['text'] = page_text
                        all_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                    
                    # Extract tables from page
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Ensure table has data
                            table_counter += 1
                            
                            # Enhanced table processing
                            table_data = self._process_table(
                                table, 
                                page_num, 
                                table_idx + 1, 
                                table_counter
                            )
                            
                            page_data['tables'].append(table_data)
                            all_tables.append(table_data)
                    
                    page_data['table_count'] = len(page_data['tables'])
                    pdf_metadata['pages'].append(page_data)
                
                # Add summary statistics
                pdf_metadata['total_tables'] = table_counter
                pdf_metadata['total_text_length'] = len(all_text)
                
                return {
                    'metadata': pdf_metadata,
                    'full_text': all_text,
                    'all_tables': all_tables
                }
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            return {
                'metadata': {'pdf_name': pdf_path.name, 'error': str(e)},
                'full_text': "",
                'all_tables': []
            }
    
    def _process_table(self, table: List[List], page_num: int, table_idx: int, global_table_id: int) -> Dict[str, Any]:
        """
        Process a single table with enhanced formatting and metadata.
        
        Args:
            table: Raw table data from pdfplumber
            page_num: Page number where table was found
            table_idx: Table index on the page
            global_table_id: Global table counter across all pages
            
        Returns:
            Processed table data with metadata
        """
        try:
            # Clean and validate table data
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cell for cell in cleaned_row):  # Skip empty rows
                    cleaned_table.append(cleaned_row)
            
            if not cleaned_table:
                return None
            
            # Extract headers and data
            headers = cleaned_table[0] if cleaned_table else []
            data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Generate formatted text representation
            formatted_text = self._format_table_as_text(headers, data_rows, page_num, table_idx)
            
            # Create table metadata
            table_metadata = {
                'table_id': f"Table_{global_table_id}",
                'page_number': page_num,
                'page_table_index': table_idx,
                'headers': headers,
                'row_count': len(data_rows),
                'column_count': len(headers),
                'formatted_text': formatted_text
            }
            
            return {
                'metadata': table_metadata,
                'dataframe': df,
                'raw_data': cleaned_table
            }
            
        except Exception as e:
            logger.error(f"Error processing table on page {page_num}: {str(e)}")
            return None
    
    def _format_table_as_text(self, headers: List[str], rows: List[List], page_num: int, table_idx: int) -> str:
        """
        Format table data as readable text with enhanced formatting.
        """
        formatted_text = f"\n--- Table {table_idx} from Page {page_num} ---\n"
        
        # Add headers
        formatted_text += "Headers: " + " | ".join(headers) + "\n"
        formatted_text += "-" * (len("Headers: " + " | ".join(headers)) + 10) + "\n"
        
        # Add rows
        for row_idx, row in enumerate(rows, 1):
            formatted_text += f"Row {row_idx}: "
            row_data = []
            for i, value in enumerate(row):
                if i < len(headers):
                    row_data.append(f"{headers[i]}: {value if value else 'N/A'}")
            formatted_text += " | ".join(row_data) + "\n"
        
        formatted_text += "\n"
        return formatted_text
    
    def save_extraction_results(self, pdf_name: str, extraction_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Save extraction results to organized file structure.
        
        Args:
            pdf_name: Name of the PDF file
            extraction_result: Result from extract_single_pdf
            
        Returns:
            Dictionary with paths to saved files
        """
        pdf_name_clean = Path(pdf_name).stem
        pdf_output_dir = self.output_dir / pdf_name_clean
        pdf_output_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save full text
            text_file = pdf_output_dir / f"{pdf_name_clean}_full_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extraction_result['full_text'])
            saved_files['text'] = str(text_file)
            
            # Save metadata
            metadata_file = pdf_output_dir / f"{pdf_name_clean}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result['metadata'], f, indent=2, default=str)
            saved_files['metadata'] = str(metadata_file)
            
            # Save tables as Excel
            if extraction_result['all_tables']:
                excel_file = pdf_output_dir / f"{pdf_name_clean}_tables.xlsx"
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    for table_data in extraction_result['all_tables']:
                        if table_data and 'dataframe' in table_data:
                            sheet_name = table_data['metadata']['table_id']
                            table_data['dataframe'].to_excel(writer, sheet_name=sheet_name, index=False)
                saved_files['tables'] = str(excel_file)
            
            # Save combined table text
            table_text_file = pdf_output_dir / f"{pdf_name_clean}_table_text.txt"
            with open(table_text_file, 'w', encoding='utf-8') as f:
                for table_data in extraction_result['all_tables']:
                    if table_data and 'metadata' in table_data:
                        f.write(table_data['metadata']['formatted_text'])
            saved_files['table_text'] = str(table_text_file)
            
            logger.info(f"Saved extraction results for {pdf_name} to {pdf_output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results for {pdf_name}: {str(e)}")
        
        return saved_files
    
    def process_multiple_pdfs(self, pdf_paths: List[str], max_workers: int = 4) -> Dict[str, Any]:
        """
        Process multiple PDFs with parallel processing.
        
        Args:
            pdf_paths: List of PDF file paths
            max_workers: Maximum number of parallel workers
            
        Returns:
            Summary of all extractions
        """
        logger.info(f"Starting batch processing of {len(pdf_paths)} PDFs")
        
        results = {}
        successful_extractions = 0
        failed_extractions = 0
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all extraction tasks
            future_to_pdf = {
                executor.submit(self.extract_single_pdf, pdf_path): pdf_path 
                for pdf_path in pdf_paths
            }
            
            # Process completed extractions
            for future in future_to_pdf:
                pdf_path = future_to_pdf[future]
                try:
                    extraction_result = future.result()
                    results[pdf_path] = extraction_result
                    
                    # Save results
                    saved_files = self.save_extraction_results(pdf_path, extraction_result)
                    results[pdf_path]['saved_files'] = saved_files
                    
                    if 'error' not in extraction_result['metadata']:
                        successful_extractions += 1
                        logger.info(f"Successfully processed: {pdf_path}")
                    else:
                        failed_extractions += 1
                        logger.error(f"Failed to process: {pdf_path}")
                        
                except Exception as e:
                    failed_extractions += 1
                    logger.error(f"Exception processing {pdf_path}: {str(e)}")
                    results[pdf_path] = {'error': str(e)}
        
        # Generate summary
        summary = {
            'total_pdfs': len(pdf_paths),
            'successful': successful_extractions,
            'failed': failed_extractions,
            'success_rate': f"{(successful_extractions/len(pdf_paths)*100):.1f}%",
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save batch summary
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing complete: {summary}")
        return results, summary

def find_pdf_files(directory: str) -> List[str]:
    """
    Find all PDF files in a directory and subdirectories.
    
    Args:
        directory: Directory to search for PDFs
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    directory_path = Path(directory)
    
    if directory_path.is_file() and directory_path.suffix.lower() == '.pdf':
        return [str(directory_path)]
    
    for pdf_file in directory_path.rglob("*.pdf"):
        pdf_files.append(str(pdf_file))
    
    return pdf_files

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = EnhancedPDFExtractor("extracted_content")
    
    # Find PDF files (you can specify a directory or individual files)
    pdf_files = find_pdf_files("/Users/ericjia/Downloads/BedsideDx")  # Change this path
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {pdf}")
        
        # Process all PDFs
        results, summary = extractor.process_multiple_pdfs(pdf_files, max_workers=2)
        
        print(f"\nProcessing Summary:")
        print(f"  Total PDFs: {summary['total_pdfs']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']}")
    else:
        print("No PDF files found in the specified directory.")
