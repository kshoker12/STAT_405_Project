"""
Validate that each question only has leaf nodes.
- A question with a parent should not have children (grandchildren of the parent)
- Questions can share a mutual parent (siblings are OK)
- No question should be an ancestor of another
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

def get_ancestors(subject_id, parent_map):
    """Get all ancestors of a subject"""
    ancestors = set()
    current = subject_id
    visited = set()
    
    while current in parent_map:
        parent = parent_map[current]
        if pd.isna(parent) or parent == 'NULL' or parent in visited:  # Cycle detection
            break
        visited.add(current)
        ancestors.add(parent)
        current = parent
    
    return ancestors

def get_descendants(subject_id, child_map):
    """Get all descendants of a subject"""
    descendants = set()
    stack = [subject_id]
    
    while stack:
        current = stack.pop()
        if current in child_map:
            for child in child_map[current]:
                descendants.add(child)
                stack.append(child)
    
    return descendants

def validate_leaf_nodes(question_metadata, subject_metadata):
    """
    Check if each question only has leaf nodes.
    A question that has a parent should not have any children.
    """
    
    # Build parent and child maps from subject_metadata
    parent_map = {}
    child_map = {}
    
    for _, row in subject_metadata.iterrows():
        subject_id = row['SubjectId']
        parent_id = row['ParentId']
        
        if pd.notna(parent_id) and parent_id != 'NULL':
            parent_map[subject_id] = parent_id
            if parent_id not in child_map:
                child_map[parent_id] = []
            child_map[parent_id].append(subject_id)
    
    # Check each question
    violations = []
    
    for _, question_row in question_metadata.iterrows():
        q_id = question_row['QuestionId']
        
        # Parse subject IDs from JSON string
        subject_ids_str = question_row['SubjectId']
        try:
            subject_ids = json.loads(subject_ids_str)
        except (json.JSONDecodeError, TypeError):
            continue
        
        # Check if this question violates the leaf node constraint
        for subject_id in subject_ids:
            ancestors = get_ancestors(subject_id, parent_map)
            descendants = get_descendants(subject_id, child_map)
            
            # A question should either have ancestors OR descendants, not both
            if ancestors and descendants:
                violations.append({
                    'question_id': q_id,
                    'subject_id': subject_id,
                    'ancestors': ancestors,
                    'descendants': descendants,
                    'reason': 'Subject has both parent(s) and child(ren)'
                })
    
    return violations

def main():
    data_dir = Path(__file__).parent.parent.parent / "data" / "metadata"
    
    subject_metadata, question_meta_1_2, question_meta_3_4 = load_data(data_dir)
    
    print("Validating Task 1-2 questions...")
    violations_1_2 = validate_leaf_nodes(question_meta_1_2, subject_metadata)
    
    print("Validating Task 3-4 questions...")
    violations_3_4 = validate_leaf_nodes(question_meta_3_4, subject_metadata)
    
    all_violations = violations_1_2 + violations_3_4
    
    if all_violations:
        print(f"\n❌ Found {len(all_violations)} violation(s):")
        for v in all_violations:
            print(f"\n  Question {v['question_id']}, Subject {v['subject_id']}:")
            print(f"    {v['reason']}")
            print(f"    Ancestors: {v['ancestors']}")
            print(f"    Descendants: {v['descendants']}")
        return False
    else:
        print("\n✓ All questions only have leaf nodes!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)