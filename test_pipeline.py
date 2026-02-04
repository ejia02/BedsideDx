"""
Test script for the Physical Exam Evidence Pipeline
Demonstrates functionality without requiring API keys
"""

import sys
import os
from excel_parser import ExcelParser

def test_excel_parsing():
    """Test Excel parsing functionality"""
    print("=" * 60)
    print("TESTING EXCEL PARSING")
    print("=" * 60)
    
    try:
        parser = ExcelParser('/Users/ericjia/Downloads/BedsideDx/AppendixChp71_Table71_1 (1).xlsx')
        documents = parser.parse_documents()
        
        print(f"✅ Successfully parsed {len(documents)} documents")
        
        # Show statistics
        chapters = set(doc.chapter for doc in documents if doc.chapter)
        ebm_boxes = set(doc.ebm_box_label for doc in documents if doc.ebm_box_label)
        maneuvers = set(doc.maneuver_base for doc in documents if doc.maneuver_base)
        
        print(f"📊 Statistics:")
        print(f"   - Chapters: {len(chapters)}")
        print(f"   - EBM Boxes: {len(ebm_boxes)}")
        print(f"   - Unique Maneuvers: {len(maneuvers)}")
        
        # Show examples by category
        categories = {}
        for doc in documents:
            if doc.ebm_box_label:
                category = doc.ebm_box_label.split()[0] if doc.ebm_box_label else "Unknown"
                if category not in categories:
                    categories[category] = []
                categories[category].append(doc)
        
        print(f"\n📋 Example documents by category:")
        for category, docs in list(categories.items())[:5]:  # Show first 5 categories
            print(f"\n   {category}:")
            for doc in docs[:2]:  # Show first 2 docs per category
                print(f"     • {doc.original_finding[:80]}...")
                print(f"       LR+: {doc.pos_lr_numeric}, LR-: {doc.neg_lr_numeric}")
        
        # Test maneuver/modifier parsing
        print(f"\n🔍 Maneuver parsing examples:")
        examples_with_modifiers = [doc for doc in documents if doc.result_modifier][:5]
        for doc in examples_with_modifiers:
            print(f"   Original: {doc.original_finding}")
            print(f"   Maneuver: {doc.maneuver_base}")
            print(f"   Modifier: {doc.result_modifier}")
            print()
        
        # Test embedding text generation
        print(f"📝 Embedding text examples:")
        for doc in documents[:3]:
            print(f"   {doc.text_for_embedding}")
            print()
        
        return documents
        
    except Exception as e:
        print(f"❌ Excel parsing failed: {e}")
        return None

def test_data_quality(documents):
    """Test data quality and completeness"""
    print("=" * 60)
    print("TESTING DATA QUALITY")
    print("=" * 60)
    
    if not documents:
        print("❌ No documents to test")
        return
    
    # Check for missing data
    missing_pos_lr = sum(1 for doc in documents if doc.pos_lr_numeric is None)
    missing_neg_lr = sum(1 for doc in documents if doc.neg_lr_numeric is None)
    missing_pretest = sum(1 for doc in documents if doc.pretest_prob_numeric is None)
    
    print(f"📊 Data completeness:")
    print(f"   - Documents with Positive LR: {len(documents) - missing_pos_lr}/{len(documents)} ({(len(documents) - missing_pos_lr)/len(documents)*100:.1f}%)")
    print(f"   - Documents with Negative LR: {len(documents) - missing_neg_lr}/{len(documents)} ({(len(documents) - missing_neg_lr)/len(documents)*100:.1f}%)")
    print(f"   - Documents with Pretest Prob: {len(documents) - missing_pretest}/{len(documents)} ({(len(documents) - missing_pretest)/len(documents)*100:.1f}%)")
    
    # Check LR ranges
    pos_lrs = [doc.pos_lr_numeric for doc in documents if doc.pos_lr_numeric is not None]
    neg_lrs = [doc.neg_lr_numeric for doc in documents if doc.neg_lr_numeric is not None]
    
    if pos_lrs:
        print(f"\n📈 Positive LR statistics:")
        print(f"   - Range: {min(pos_lrs):.2f} to {max(pos_lrs):.2f}")
        print(f"   - Mean: {sum(pos_lrs)/len(pos_lrs):.2f}")
        print(f"   - High LR (>10): {sum(1 for lr in pos_lrs if lr > 10)} documents")
    
    if neg_lrs:
        print(f"\n📉 Negative LR statistics:")
        print(f"   - Range: {min(neg_lrs):.2f} to {max(neg_lrs):.2f}")
        print(f"   - Mean: {sum(neg_lrs)/len(neg_lrs):.2f}")
        print(f"   - Low LR (<0.1): {sum(1 for lr in neg_lrs if lr < 0.1)} documents")

def test_search_simulation(documents):
    """Simulate search functionality without actual vector search"""
    print("=" * 60)
    print("TESTING SEARCH SIMULATION")
    print("=" * 60)
    
    if not documents:
        print("❌ No documents to search")
        return
    
    # Simulate text-based search
    test_queries = [
        "heart murmur",
        "aortic stenosis", 
        "blood pressure",
        "chest pain",
        "shortness of breath"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Simulating search for: '{query}'")
        
        # Simple text matching
        matches = []
        for doc in documents:
            score = 0
            query_words = query.lower().split()
            
            # Check in various fields
            text_fields = [
                doc.text_for_embedding.lower(),
                doc.original_finding.lower(),
                doc.ebm_box_label.lower(),
                doc.maneuver_base.lower()
            ]
            
            for field in text_fields:
                for word in query_words:
                    if word in field:
                        score += 1
            
            if score > 0:
                matches.append((doc, score))
        
        # Sort by score and show top results
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:3]
        
        print(f"   Found {len(matches)} matches, showing top {len(top_matches)}:")
        for i, (doc, score) in enumerate(top_matches, 1):
            print(f"   {i}. {doc.original_finding[:60]}...")
            print(f"      EBM Box: {doc.ebm_box_label}")
            print(f"      LR+: {doc.pos_lr_numeric}, LR-: {doc.neg_lr_numeric}")
            print(f"      Match Score: {score}")
            print()

def show_setup_instructions():
    """Show setup instructions for full functionality"""
    print("=" * 60)
    print("SETUP INSTRUCTIONS FOR FULL FUNCTIONALITY")
    print("=" * 60)
    
    print("To use the complete vector search pipeline:")
    print()
    print("1. 📝 Create environment file:")
    print("   cp env_example.txt .env")
    print("   # Edit .env with your API keys")
    print()
    print("2. 🔧 Set up MongoDB Atlas:")
    print("   # Follow instructions in MONGODB_SETUP.md")
    print()
    print("3. 🚀 Run full pipeline:")
    print("   python physical_exam_pipeline.py full --clear")
    print()
    print("4. 🔍 Search the database:")
    print("   python physical_exam_pipeline.py search --query 'heart murmur'")
    print()
    print("📚 See README_VECTOR_PIPELINE.md for complete documentation")

def main():
    """Run all tests"""
    print("🧪 PHYSICAL EXAM EVIDENCE PIPELINE TEST SUITE")
    print("=" * 60)
    
    # Test Excel parsing
    documents = test_excel_parsing()
    
    if documents:
        # Test data quality
        test_data_quality(documents)
        
        # Test search simulation
        test_search_simulation(documents)
    
    # Show setup instructions
    show_setup_instructions()
    
    print("\n✅ Test suite completed!")
    print("The pipeline is ready for use with proper API keys and MongoDB setup.")

if __name__ == "__main__":
    main()

