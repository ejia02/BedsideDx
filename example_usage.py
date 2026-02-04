#!/usr/bin/env python3
"""
Example usage of the multi-PDF extractor.
This shows how to use the enhanced PDF extraction for multiple files.
"""

from multi_pdf_extract import MultiPDFExtractor, find_pdfs
from pathlib import Path

def main():
    # Initialize the extractor
    extractor = MultiPDFExtractor("extracted_content")
    
    # Option 1: Process all PDFs in a directory
    print("=== Option 1: Process all PDFs in directory ===")
    pdf_directory = "/Users/ericjia/Downloads/BedsideDx"  # Change this to your PDF directory
    pdf_files = find_pdfs(pdf_directory)
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {Path(pdf).name}")
        
        # Process all PDFs
        results = extractor.process_multiple_pdfs(pdf_files)
        
        # Print summary of what was extracted
        for pdf_path, result in results.items():
            if 'error' not in result:
                print(f"\n{Path(pdf_path).name}:")
                print(f"  - Pages: {result['total_pages']}")
                print(f"  - Tables: {result['total_tables']}")
                print(f"  - Files saved: {len(result.get('saved_files', {}))}")
            else:
                print(f"\n{Path(pdf_path).name}: ERROR - {result['error']}")
    
    # Option 2: Process specific PDF files
    print("\n=== Option 2: Process specific PDF files ===")
    specific_pdfs = [
        "/path/to/your/pdf1.pdf",
        "/path/to/your/pdf2.pdf",
        # Add more specific PDF paths here
    ]
    
    # Filter to only existing files
    existing_pdfs = [pdf for pdf in specific_pdfs if Path(pdf).exists()]
    
    if existing_pdfs:
        results = extractor.process_multiple_pdfs(existing_pdfs)
    else:
        print("No existing PDF files found in the specific list.")
    
    # Option 3: Process a single PDF (like the original)
    print("\n=== Option 3: Process single PDF ===")
    single_pdf = "/Users/ericjia/Downloads/BedsideDx/COPD_Mcgee.pdf"  # Change this path
    
    if Path(single_pdf).exists():
        result = extractor.extract_pdf_with_context(single_pdf)
        if 'error' not in result:
            saved_files = extractor.save_results(single_pdf, result)
            print(f"Single PDF processed successfully!")
            print(f"Files saved: {list(saved_files.keys())}")
        else:
            print(f"Error processing single PDF: {result['error']}")
    else:
        print(f"PDF file not found: {single_pdf}")

if __name__ == "__main__":
    main()
