"""
Excel Parser for Physical Exam Evidence Table
Parses AppendixChp71_Table71_1.xlsx into structured documents for vector search
"""

import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExamDocument:
    """Structured document representing one exam maneuver result"""
    # Metadata
    chapter: str
    ebm_box_id: str
    ebm_box_label: str
    maneuver_base: str
    result_modifier: Optional[str]
    
    # Raw metrics
    pretest_prob_raw: str
    pos_lr_raw: str
    neg_lr_raw: str
    
    # Parsed numeric values
    pos_lr_numeric: Optional[float]
    neg_lr_numeric: Optional[float]
    pretest_prob_numeric: Optional[float]
    
    # Text for embedding
    text_for_embedding: str
    
    # Full finding text
    original_finding: str


class ExcelParser:
    """Parser for physical exam evidence Excel file"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = None
        self.documents = []
        
    def load_excel(self) -> None:
        """Load Excel file into pandas DataFrame"""
        try:
            self.df = pd.read_excel(self.excel_path, header=None)
            logger.info(f"Loaded Excel file with shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise
    
    def parse_lr_value(self, lr_string: str) -> Optional[float]:
        """Extract first numeric value from LR string"""
        if pd.isna(lr_string) or lr_string == '…' or lr_string == 'NS':
            return None
        
        # Look for first number (including decimals)
        match = re.search(r'(\d+\.?\d*)', str(lr_string))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def parse_pretest_prob(self, pretest_string: str) -> Optional[float]:
        """Extract numeric pretest probability (take first number if range)"""
        if pd.isna(pretest_string):
            return None
        
        # Look for first number
        match = re.search(r'(\d+)', str(pretest_string))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def split_maneuver_and_modifier(self, finding: str) -> Tuple[str, Optional[str]]:
        """Split finding into base maneuver and result modifier"""
        if pd.isna(finding):
            return "", None
        
        finding = str(finding).strip()
        
        # Look for comparators and digits
        # Pattern: capture everything before a digit or comparator
        pattern = r'^(.*?)(?=\s*(?:[<>≥≤=]\s*\d+|\d+))'
        match = re.search(pattern, finding)
        
        if match:
            base = match.group(1).strip()
            modifier = finding[len(base):].strip()
            return base, modifier if modifier else None
        else:
            # No clear separator found, return whole string as base
            return finding, None
    
    def create_embedding_text(self, doc: ExamDocument) -> str:
        """Create natural language text for embedding"""
        parts = []
        
        # Add diagnosis
        if doc.ebm_box_label:
            parts.append(f"Diagnosis: {doc.ebm_box_label}")
        
        # Add maneuver
        if doc.maneuver_base:
            parts.append(f"Maneuver: {doc.maneuver_base}")
        
        # Add result modifier
        if doc.result_modifier:
            parts.append(f"Result: {doc.result_modifier}")
        
        # Add LR information
        lr_info = []
        if doc.pos_lr_numeric:
            lr_info.append(f"Positive LR: {doc.pos_lr_numeric}")
        if doc.neg_lr_numeric:
            lr_info.append(f"Negative LR: {doc.neg_lr_numeric}")
        
        if lr_info:
            parts.append(", ".join(lr_info))
        
        # Add pretest probability
        if doc.pretest_prob_numeric:
            parts.append(f"Pretest probability: {doc.pretest_prob_numeric}%")
        
        return ". ".join(parts)
    
    def parse_documents(self) -> List[ExamDocument]:
        """Parse Excel file into structured documents"""
        if self.df is None:
            self.load_excel()
        
        documents = []
        current_chapter = ""
        current_ebm_box_id = ""
        current_ebm_box_label = ""
        
        for i, row in self.df.iterrows():
            cell0 = str(row[0]) if pd.notna(row[0]) else ''
            
            # Skip header rows
            if cell0.lower().startswith('finding') or cell0 == 'nan':
                continue
            
            # Update chapter
            if cell0.startswith('Chapter'):
                current_chapter = cell0
                logger.info(f"Processing {current_chapter}")
                continue
            
            # Update EBM box
            if cell0.startswith('EBM Box'):
                # Parse EBM box ID and label
                match = re.match(r'EBM Box ([\d.]+)\s+(.*)', cell0)
                if match:
                    current_ebm_box_id = match.group(1)
                    current_ebm_box_label = match.group(2)
                else:
                    current_ebm_box_id = cell0
                    current_ebm_box_label = cell0
                logger.info(f"Processing {cell0}")
                continue
            
            # Skip rows not under an EBM box
            if not current_ebm_box_id:
                continue
            
            # Parse data row
            finding = str(row[0]) if pd.notna(row[0]) else ''
            pos_lr_raw = str(row[1]) if pd.notna(row[1]) else ''
            neg_lr_raw = str(row[2]) if pd.notna(row[2]) else ''
            pretest_prob_raw = str(row[3]) if pd.notna(row[3]) else ''
            
            # Skip empty findings
            if not finding or finding == 'nan':
                continue
            
            # Parse maneuver and modifier
            maneuver_base, result_modifier = self.split_maneuver_and_modifier(finding)
            
            # Parse numeric values
            pos_lr_numeric = self.parse_lr_value(pos_lr_raw)
            neg_lr_numeric = self.parse_lr_value(neg_lr_raw)
            pretest_prob_numeric = self.parse_pretest_prob(pretest_prob_raw)
            
            # Create document
            doc = ExamDocument(
                chapter=current_chapter,
                ebm_box_id=current_ebm_box_id,
                ebm_box_label=current_ebm_box_label,
                maneuver_base=maneuver_base,
                result_modifier=result_modifier,
                pretest_prob_raw=pretest_prob_raw,
                pos_lr_raw=pos_lr_raw,
                neg_lr_raw=neg_lr_raw,
                pos_lr_numeric=pos_lr_numeric,
                neg_lr_numeric=neg_lr_numeric,
                pretest_prob_numeric=pretest_prob_numeric,
                text_for_embedding="",  # Will be filled below
                original_finding=finding
            )
            
            # Generate embedding text
            doc.text_for_embedding = self.create_embedding_text(doc)
            
            documents.append(doc)
        
        logger.info(f"Parsed {len(documents)} documents")
        return documents


def main():
    """Test the parser"""
    parser = ExcelParser('/Users/ericjia/Downloads/BedsideDx/AppendixChp71_Table71_1 (1).xlsx')
    documents = parser.parse_documents()
    
    # Print first few documents
    for i, doc in enumerate(documents[:5]):
        print(f"\n--- Document {i+1} ---")
        print(f"Chapter: {doc.chapter}")
        print(f"EBM Box: {doc.ebm_box_id} - {doc.ebm_box_label}")
        print(f"Maneuver: {doc.maneuver_base}")
        print(f"Result Modifier: {doc.result_modifier}")
        print(f"Pos LR: {doc.pos_lr_numeric} (raw: {doc.pos_lr_raw})")
        print(f"Neg LR: {doc.neg_lr_numeric} (raw: {doc.neg_lr_raw})")
        print(f"Pretest Prob: {doc.pretest_prob_numeric}% (raw: {doc.pretest_prob_raw})")
        print(f"Embedding Text: {doc.text_for_embedding}")
        print(f"Original: {doc.original_finding}")


if __name__ == "__main__":
    main()

