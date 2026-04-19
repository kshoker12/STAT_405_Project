"""
Check if each question in question_metadata only has leaf nodes in the subject hierarchy.
A leaf node is a subject that does not have any children in the hierarchy.
"""

import pandas as pd
import sys
import json
from pathlib import Path

def load_data(data_dir):
    """Load metadata files"""
    subject_metadata = pd.read_csv(data_dir / "subject_metadata.csv")
    question_meta_1_2 = pd.read_csv(data_dir / "question_metadata_task_1_2.csv")
    question_meta_3_4 = pd.read_csv(data_dir / "question_metadata_task_3_4.csv")
    
    return subject_metadata, question_meta_1_2, question_meta_3_4

def identify_leaf_nodes(subject_metadata):
    """Identify all leaf nodes (subjects with no children)"""
    # Get all subject IDs
    all_subjects = set(subject_metadata['SubjectId'].values)
    
    # Get all subjects that are parents (have children)
    parent_subjects = set(subject_metadata[pd.notna(subject_metadata['ParentId']) & 
                                         (subject_metadata['ParentId'] != 'NULL')]['ParentId'].values)
    
    # Leaf nodes are those that are never parents
    leaf_nodes = all_subjects - parent_subjects
    
    return leaf_nodes

def validate_questions_have_leaf_nodes(question_metadata, leaf_nodes):
    """
    Check if each question only uses leaf nodes.
    Returns violations where a question uses non-leaf nodes.
    """
    violations = []
    
    for _, question_row in question_metadata.iterrows():
        q_id = question_row['QuestionId']
        
        # Parse subject IDs from JSON string
        subject_ids_str = question_row['SubjectId']
        try:
            subject_ids = json.loads(subject_ids_str)
        except (json.JSONDecodeError, TypeError):
            continue
        
        # Check if any subject ID is not a leaf node
        non_leaf_subjects = [s for s in subject_ids if s not in leaf_nodes]
        
        if non_leaf_subjects:
            violations.append({
                'question_id': q_id,
                'non_leaf_subjects': non_leaf_subjects,
                'all_subjects': subject_ids,
                'reason': f'Question uses {len(non_leaf_subjects)} non-leaf node(s)'
            })
    
    return violations

def main():
    data_dir = Path(__file__).parent.parent.parent / "data" / "metadata"
    
    # Load data
    subject_metadata, question_meta_1_2, question_meta_3_4 = load_data(data_dir)
    
    # Identify leaf nodes
    leaf_nodes = identify_leaf_nodes(subject_metadata)
    print(f"Found {len(leaf_nodes)} leaf nodes out of {len(subject_metadata)} total subjects")
    print(f"Leaf nodes: {sorted(leaf_nodes)}\n")
    
    # Validate Task 1-2 questions
    print("Validating Task 1-2 questions...")
    violations_1_2 = validate_questions_have_leaf_nodes(question_meta_1_2, leaf_nodes)
    
    # Validate Task 3-4 questions
    print("Validating Task 3-4 questions...")
    violations_3_4 = validate_questions_have_leaf_nodes(question_meta_3_4, leaf_nodes)
    
    all_violations = violations_1_2 + violations_3_4
    
    if all_violations:
        print(f"\n❌ Found {len(all_violations)} question(s) with non-leaf node violation(s):")
        
        # Group by question for cleaner output
        for v in all_violations[:20]:  # Show first 20
            print(f"\n  Question {v['question_id']}:")
            print(f"    {v['reason']}")
            print(f"    Non-leaf subjects: {v['non_leaf_subjects']}")
            print(f"    All subjects: {v['all_subjects']}")
        
        if len(all_violations) > 20:
            print(f"\n  ... and {len(all_violations) - 20} more violations")
        
        return False
    else:
        print("\n✓ All questions only use leaf nodes!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)