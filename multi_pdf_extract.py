import pdfplumber
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

class MultiPDFExtractor:
    """
    Multi-PDF extractor that processes multiple PDFs and handles tables/text together.
    """
    
    def __init__(self, output_dir: str = "extracted_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_pdf_with_context(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and tables from a PDF, maintaining context between them.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        pdf_path = Path(pdf_path)
        print(f"Processing: {pdf_path.name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Store all content with page context
                all_content = []
                all_tables = []
                table_counter = 0
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_content = {
                        'page_number': page_num,
                        'text': "",
                        'tables': []
                    }
                    
                    # Extract text from page
                    page_text = page.extract_text()
                    if page_text:
                        page_content['text'] = page_text
                    
                    # Extract tables from page
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:
                            table_counter += 1
                            
                            # Process table
                            processed_table = self._process_table(table, page_num, table_counter)
                            if processed_table:
                                page_content['tables'].append(processed_table)
                                all_tables.append(processed_table)
                    
                    all_content.append(page_content)
                
                return {
                    'pdf_name': pdf_path.name,
                    'total_pages': len(pdf.pages),
                    'total_tables': table_counter,
                    'content_by_page': all_content,
                    'all_tables': all_tables,
                    'extraction_time': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")
            return {'pdf_name': pdf_path.name, 'error': str(e)}
    
    def _process_table(self, table: List[List], page_num: int, table_id: int) -> Dict[str, Any]:
        """Process a single table with proper formatting."""
        try:
            # Clean table data
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cell for cell in cleaned_row):  # Skip empty rows
                    cleaned_table.append(cleaned_row)
            
            if not cleaned_table:
                return None
            
            # Extract headers and data
            headers = cleaned_table[0]
            data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            return {
                'table_id': f"Table_{table_id}",
                'page_number': page_num,
                'headers': headers,
                'data': data_rows,
                'dataframe': df,
                'row_count': len(data_rows),
                'column_count': len(headers)
            }
            
        except Exception as e:
            print(f"Error processing table on page {page_num}: {str(e)}")
            return None
    
    def save_results(self, pdf_name: str, extraction_result: Dict[str, Any]) -> Dict[str, str]:
        """Save extraction results to organized files."""
        pdf_name_clean = Path(pdf_name).stem
        pdf_output_dir = self.output_dir / pdf_name_clean
        pdf_output_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        try:
            # 1. Save full text with page breaks
            text_file = pdf_output_dir / f"{pdf_name_clean}_full_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for page_content in extraction_result['content_by_page']:
                    f.write(f"\n--- Page {page_content['page_number']} ---\n")
                    f.write(page_content['text'])
                    f.write("\n")
            saved_files['text'] = str(text_file)
            
            # 2. Save tables as Excel with multiple sheets
            if extraction_result['all_tables']:
                excel_file = pdf_output_dir / f"{pdf_name_clean}_tables.xlsx"
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    for table in extraction_result['all_tables']:
                        table['dataframe'].to_excel(
                            writer, 
                            sheet_name=table['table_id'], 
                            index=False
                        )
                saved_files['tables'] = str(excel_file)
            
            # 3. Save combined text and tables in context
            combined_file = pdf_output_dir / f"{pdf_name_clean}_combined.txt"
            with open(combined_file, 'w', encoding='utf-8') as f:
                for page_content in extraction_result['content_by_page']:
                    f.write(f"\n=== PAGE {page_content['page_number']} ===\n\n")
                    
                    # Write page text
                    if page_content['text']:
                        f.write("TEXT:\n")
                        f.write(page_content['text'])
                        f.write("\n\n")
                    
                    # Write tables from this page
                    if page_content['tables']:
                        f.write("TABLES:\n")
                        for table in page_content['tables']:
                            f.write(f"\n--- {table['table_id']} ---\n")
                            f.write("Headers: " + " | ".join(table['headers']) + "\n")
                            f.write("-" * 50 + "\n")
                            for row in table['data']:
                                f.write(" | ".join(row) + "\n")
                            f.write("\n")
            
            saved_files['combined'] = str(combined_file)
            
            # 4. Save metadata
            metadata_file = pdf_output_dir / f"{pdf_name_clean}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, default=str)
            saved_files['metadata'] = str(metadata_file)
            
            print(f"Saved results for {pdf_name} to {pdf_output_dir}")
            
        except Exception as e:
            print(f"Error saving results for {pdf_name}: {str(e)}")
        
        return saved_files
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Process multiple PDFs and return results."""
        results = {}
        successful = 0
        failed = 0
        
        for pdf_path in pdf_paths:
            print(f"\nProcessing: {pdf_path}")
            
            # Extract content
            extraction_result = self.extract_pdf_with_context(pdf_path)
            
            if 'error' not in extraction_result:
                # Save results
                saved_files = self.save_results(pdf_path, extraction_result)
                extraction_result['saved_files'] = saved_files
                successful += 1
            else:
                failed += 1
            
            results[pdf_path] = extraction_result
        
        # Print summary
        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Total PDFs: {len(pdf_paths)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(successful/len(pdf_paths)*100):.1f}%")
        
        return results

def find_pdfs(directory: str) -> List[str]:
    """Find all PDF files in directory."""
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
    extractor = MultiPDFExtractor("extracted_content")
    
    # Find PDF files
    pdf_files = find_pdfs("/Users/ericjia/Downloads/BedsideDx")  # Change this path
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {Path(pdf).name}")
        
        # Process all PDFs
        results = extractor.process_multiple_pdfs(pdf_files)
        
        print(f"\nResults saved to: {extractor.output_dir}")
    else:
        print("No PDF files found.")


// think about how to rank the symptoms or suggestions 
// think about how to systemically design the pipeline
// scrape Stanford 2025 website for all the physical exam maneuvers
// scrape text from textbook