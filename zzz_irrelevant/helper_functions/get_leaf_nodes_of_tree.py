"""
Get the leaf nodes of a tree structure from a list of SubjectIds.
This function:
1. Validates that the given SubjectIds form a valid tree structure
2. Returns the leaf nodes (subjects with no children)
3. Raises an error if they don't form a tree
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Set

def load_subject_metadata(data_dir=None):
    """Load subject metadata CSV file"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "metadata"
    
    return pd.read_csv(data_dir / "subject_metadata.csv")

def get_ancestors(subject_id, parent_map):
    """Get all ancestors of a subject"""
    ancestors = []
    current = subject_id
    visited = set()
    
    while current in parent_map:
        parent = parent_map[current]
        if pd.isna(parent) or parent == 'NULL' or parent in visited:
            break
        visited.add(current)
        ancestors.append(parent)
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

def is_tree_structure(subject_ids, subject_metadata):
    """
    Validate that the given subject_ids form a tree structure.
    A tree must have:
    1. A single root (subject with no parent in the subset)
    2. All subjects connected through parent-child relationships
    3. No cycles
    """
    
    if not subject_ids:
        raise ValueError("subject_ids list cannot be empty")
    
    subject_ids_set = set(subject_ids)
    
    # Build parent and child maps
    parent_map = {}
    child_map = {}
    
    for _, row in subject_metadata.iterrows():
        subject_id = row['SubjectId']
        parent_id = row['ParentId']
        
        if subject_id in subject_ids_set:
            if pd.notna(parent_id) and parent_id != 'NULL':
                parent_map[subject_id] = parent_id
                if parent_id not in child_map:
                    child_map[parent_id] = []
                child_map[parent_id].append(subject_id)
            else:
                # Root node (no parent)
                if subject_id not in child_map:
                    child_map[subject_id] = []
    
    # Find root nodes (subjects with no parent in the subset)
    roots = [s for s in subject_ids_set if s not in parent_map]
    
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root. Found {len(roots)}: {roots}")
    
    # Check connectivity: all subjects must be reachable from root
    root = roots[0]
    visited = set()
    stack = [root]
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        
        if current in child_map:
            for child in child_map[current]:
                if child not in visited:
                    stack.append(child)
    
    if visited != subject_ids_set:
        disconnected = subject_ids_set - visited
        raise ValueError(f"Not all subjects are connected to the root. "
                        f"Disconnected subjects: {disconnected}")
    
    # Check for cycles (shouldn't happen in a proper parent-child relationship)
    for subject_id in subject_ids_set:
        visited_ancestors = set()
        current = subject_id
        
        while current in parent_map:
            parent = parent_map[current]
            if parent in visited_ancestors:
                raise ValueError(f"Cycle detected involving subject {subject_id}")
            visited_ancestors.add(parent)
            current = parent
    
    return True

def get_leaf_nodes(subject_ids, data_dir=None):
    """
    Get the leaf nodes from a tree structure defined by subject_ids.
    
    Parameters:
    -----------
    subject_ids : List[int]
        List of SubjectIds that form a tree
    data_dir : Path, optional
        Directory containing subject_metadata.csv
    
    Returns:
    --------
    List[int]
        List of leaf node SubjectIds (subjects with no children)
    
    Raises:
    -------
    ValueError
        If the subject_ids do not form a valid tree structure
    """
    
    subject_metadata = load_subject_metadata(data_dir)
    
    # Validate that it's a tree
    is_tree_structure(subject_ids, subject_metadata)
    
    subject_ids_set = set(subject_ids)
    
    # Find all subjects that are parents (have children)
    parent_subjects = set()
    
    for _, row in subject_metadata.iterrows():
        subject_id = row['SubjectId']
        parent_id = row['ParentId']
        
        if subject_id in subject_ids_set and pd.notna(parent_id) and parent_id != 'NULL':
            parent_id_val = int(parent_id) if isinstance(parent_id, float) else parent_id
            if parent_id_val in subject_ids_set:
                parent_subjects.add(parent_id_val)
    
    # Leaf nodes are subjects that are never parents
    leaf_nodes = [s for s in subject_ids if s not in parent_subjects]
    
    return sorted(leaf_nodes)


def validate_questions_tree_structure(data_dir=None):
    """
    Check if each question from question_metadata files fits the tree structure.
    
    Parameters:
    -----------
    data_dir : Path, optional
        Directory containing the metadata CSV files
    
    Returns:
    --------
    dict
        Contains:
        - 'valid_questions': List of questions that form valid trees
        - 'invalid_questions': List of dicts with question_id, error, and subject_ids
        - 'summary': Summary statistics
    """
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "metadata"
    
    subject_metadata = load_subject_metadata(data_dir)
    
    # Pre-build parent and child maps for efficiency
    parent_map = {}
    child_map = {}
    all_subjects = set()
    
    for _, row in subject_metadata.iterrows():
        subject_id = row['SubjectId']
        parent_id = row['ParentId']
        all_subjects.add(subject_id)
        
        if pd.notna(parent_id) and parent_id != 'NULL':
            parent_map[subject_id] = parent_id
            if parent_id not in child_map:
                child_map[parent_id] = []
            child_map[parent_id].append(subject_id)
    
    # Load both question metadata files
    question_meta_1_2 = pd.read_csv(data_dir / "question_metadata_task_1_2.csv")
    question_meta_3_4 = pd.read_csv(data_dir / "question_metadata_task_3_4.csv")
    
    # Combine both datasets
    question_metadata = pd.concat([question_meta_1_2, question_meta_3_4], ignore_index=True)
    
    valid_questions = []
    invalid_questions = []
    
    for _, question_row in question_metadata.iterrows():
        q_id = question_row['QuestionId']
        subject_ids_str = question_row['SubjectId']
        
        # Parse subject IDs from JSON string
        try:
            subject_ids = json.loads(subject_ids_str)
        except (json.JSONDecodeError, TypeError):
            invalid_questions.append({
                'question_id': q_id,
                'error': 'Could not parse SubjectId JSON',
                'subject_ids': subject_ids_str
            })
            continue
        
        # Validate tree structure
        subject_ids_set = set(subject_ids)
        
        # Find root nodes (subjects with no parent in the subset)
        roots = [s for s in subject_ids_set if s not in parent_map]
        
        try:
            if len(roots) != 1:
                raise ValueError(f"Tree must have exactly one root. Found {len(roots)}: {roots}")
            
            # Check connectivity
            root = roots[0]
            visited = set()
            stack = [root]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                if current in child_map:
                    for child in child_map[current]:
                        if child in subject_ids_set and child not in visited:
                            stack.append(child)
            
            if visited != subject_ids_set:
                disconnected = subject_ids_set - visited
                raise ValueError(f"Disconnected subjects: {disconnected}")
            
            # Find leaf nodes
            parent_subjects = set()
            for subject_id in subject_ids_set:
                if subject_id in child_map:
                    for child in child_map[subject_id]:
                        if child in subject_ids_set:
                            parent_subjects.add(subject_id)
            
            leaf_nodes = sorted([s for s in subject_ids if s not in parent_subjects])
            
            valid_questions.append({
                'question_id': q_id,
                'subject_ids': subject_ids,
                'leaf_nodes': leaf_nodes
            })
        except ValueError as e:
            invalid_questions.append({
                'question_id': q_id,
                'error': str(e),
                'subject_ids': subject_ids
            })
    
    # Create summary
    summary = {
        'total_questions': len(question_metadata),
        'valid_trees': len(valid_questions),
        'invalid_trees': len(invalid_questions),
        'valid_percentage': (len(valid_questions) / len(question_metadata) * 100) if len(question_metadata) > 0 else 0
    }
    
    return {
        'valid_questions': valid_questions,
        'invalid_questions': invalid_questions,
        'summary': summary
    }


def validate_forest_structure(subject_ids, valid_roots, subject_metadata):
    """
    Validate that subject_ids form a valid forest (collection of trees).
    
    Parameters:
    -----------
    subject_ids : List[int]
        List of SubjectIds
    valid_roots : List[int]
        List of valid root nodes for the forest (e.g., [3, 1642])
    subject_metadata : pd.DataFrame
        Subject metadata
    
    Returns:
    --------
    dict
        Contains validation results with forest structure details
    
    Raises:
    -------
    ValueError
        If the structure is not a valid forest
    """
    
    if not subject_ids:
        raise ValueError("subject_ids list cannot be empty")
    
    subject_ids_set = set(subject_ids)
    
    # Build parent and child maps
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
        else:
            if subject_id not in child_map:
                child_map[subject_id] = []
    
    # Find all root nodes (subjects with no parent in the subset)
    roots_in_subset = [s for s in subject_ids_set if s not in parent_map]
    
    # Validate forest constraints
    if len(roots_in_subset) > 2:
        raise ValueError(f"Forest can have at most 2 trees. Found {len(roots_in_subset)}: {roots_in_subset}")
    
    # Check that all roots are valid
    invalid_roots = [r for r in roots_in_subset if r not in valid_roots]
    if invalid_roots:
        raise ValueError(f"Invalid roots: {invalid_roots}. Must be in {valid_roots}")
    
    # Check connectivity: all subjects must be reachable from one of the roots
    visited = set()
    trees = {}
    
    for root in roots_in_subset:
        stack = [root]
        tree_members = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            tree_members.add(current)
            
            if current in child_map:
                for child in child_map[current]:
                    if child in subject_ids_set and child not in visited:
                        stack.append(child)
        
        trees[root] = sorted(list(tree_members))
    
    if visited != subject_ids_set:
        disconnected = subject_ids_set - visited
        raise ValueError(f"Disconnected subjects: {disconnected}")
    
    # Get leaf nodes for each tree
    forest_structure = {}
    for root, tree_members in trees.items():
        parent_subjects = set()
        for subject_id in tree_members:
            if subject_id in child_map:
                for child in child_map[subject_id]:
                    if child in tree_members:
                        parent_subjects.add(subject_id)
        
        leaf_nodes = sorted([s for s in tree_members if s not in parent_subjects])
        forest_structure[root] = {
            'members': tree_members,
            'leaf_nodes': leaf_nodes,
            'num_nodes': len(tree_members),
            'num_leaves': len(leaf_nodes)
        }
    
    return {
        'is_valid_forest': True,
        'num_trees': len(roots_in_subset),
        'roots': roots_in_subset,
        'forest_structure': forest_structure
    }


def validate_forest_questions(invalid_questions, subject_metadata, valid_roots=[3, 1642]):
    """
    Check if invalid questions form valid forests.
    
    Parameters:
    -----------
    invalid_questions : List[dict]
        List of invalid questions from validation
    subject_metadata : pd.DataFrame
        Subject metadata
    valid_roots : List[int]
        Valid root nodes for the forest
    
    Returns:
    --------
    dict
        Classification of questions as valid forests or invalid
    """
    
    valid_forests = []
    still_invalid = []
    
    for q in invalid_questions:
        q_id = q['question_id']
        subject_ids = q['subject_ids']
        
        try:
            forest_result = validate_forest_structure(
                subject_ids, 
                valid_roots, 
                subject_metadata
            )
            valid_forests.append({
                'question_id': q_id,
                'subject_ids': subject_ids,
                'forest': forest_result
            })
        except ValueError as e:
            still_invalid.append({
                'question_id': q_id,
                'error': str(e),
                'subject_ids': subject_ids
            })
    
    return {
        'valid_forests': valid_forests,
        'still_invalid': still_invalid,
        'summary': {
            'total': len(invalid_questions),
            'valid_forests': len(valid_forests),
            'still_invalid': len(still_invalid)
        }
    }


def print_validation_report(validation_result, show_valid=False, max_invalid=20):
    """
    Pretty print the validation results.
    
    Parameters:
    -----------
    validation_result : dict
        Result from validate_questions_tree_structure()
    show_valid : bool
        Whether to show valid questions (default: False)
    max_invalid : int
        Maximum number of invalid questions to display (default: 20)
    """
    
    summary = validation_result['summary']
    print("=" * 80)
    print("QUESTION TREE STRUCTURE VALIDATION REPORT")
    print("=" * 80)
    print(f"\nTotal questions: {summary['total_questions']}")
    print(f"Valid trees: {summary['valid_trees']}")
    print(f"Invalid trees: {summary['invalid_trees']}")
    print(f"Valid percentage: {summary['valid_percentage']:.2f}%\n")
    
    if show_valid and validation_result['valid_questions']:
        print("\n" + "=" * 80)
        print("VALID QUESTIONS (sample):")
        print("=" * 80)
        for q in validation_result['valid_questions'][:5]:
            print(f"\nQuestion {q['question_id']}:")
            print(f"  Subjects: {q['subject_ids']}")
            print(f"  Leaf nodes: {q['leaf_nodes']}")
    
    if validation_result['invalid_questions']:
        print("\n" + "=" * 80)
        print("INVALID QUESTIONS:")
        print("=" * 80)
        
        invalid_list = validation_result['invalid_questions'][:max_invalid]
        for q in invalid_list:
            print(f"\nQuestion {q['question_id']}:")
            print(f"  Error: {q['error']}")
            print(f"  Subjects: {q['subject_ids']}")
        
        if len(validation_result['invalid_questions']) > max_invalid:
            print(f"\n... and {len(validation_result['invalid_questions']) - max_invalid} more invalid questions")


def print_forest_validation_report(forest_result, max_display=5):
    """Pretty print the forest validation results"""
    summary = forest_result['summary']
    print("\n" + "=" * 80)
    print("FOREST VALIDATION REPORT")
    print("=" * 80)
    print(f"\nTotal previously invalid questions: {summary['total']}")
    print(f"Valid forests: {summary['valid_forests']}")
    print(f"Still invalid: {summary['still_invalid']}\n")
    
    if forest_result['valid_forests']:
        print("=" * 80)
        print("VALID FORESTS:")
        print("=" * 80)
        
        for q in forest_result['valid_forests'][:max_display]:
            print(f"\nQuestion {q['question_id']}:")
            forest = q['forest']
            print(f"  Number of trees: {forest['num_trees']}")
            print(f"  Root nodes: {forest['roots']}")
            
            for root, tree_info in forest['forest_structure'].items():
                print(f"\n  Tree rooted at {root}:")
                print(f"    Members: {tree_info['members']}")
                print(f"    Leaf nodes: {tree_info['leaf_nodes']}")
                print(f"    Nodes: {tree_info['num_nodes']}, Leaves: {tree_info['num_leaves']}")
        
        if len(forest_result['valid_forests']) > max_display:
            print(f"\n... and {len(forest_result['valid_forests']) - max_display} more valid forests")
    
    if forest_result['still_invalid']:
        print("\n" + "=" * 80)
        print("STILL INVALID:")
        print("=" * 80)
        
        for q in forest_result['still_invalid'][:max_display]:
            print(f"\nQuestion {q['question_id']}:")
            print(f"  Error: {q['error']}")
            print(f"  Subjects: {q['subject_ids']}")


if __name__ == "__main__":
    # Example 1: Basic tree validation
    print("EXAMPLE 1: Basic Tree Validation")
    print("-" * 50)
    try:
        test_subjects = [3, 32, 35, 36, 39]
        leaf_nodes = get_leaf_nodes(test_subjects)
        print(f"Tree with subjects {test_subjects}")
        print(f"Leaf nodes: {leaf_nodes}\n")
        
        test_subjects_2 = [3, 49]
        leaf_nodes_2 = get_leaf_nodes(test_subjects_2)
        print(f"Tree with subjects {test_subjects_2}")
        print(f"Leaf nodes: {leaf_nodes_2}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: Validate all questions in metadata
    print("\n\nEXAMPLE 2: Validate All Questions (Tree Structure)")
    print("-" * 50)
    result = validate_questions_tree_structure()
    print_validation_report(result, show_valid=False, max_invalid=5)
    
    # Example 3: Check if invalid questions form valid forests
    print("\n\nEXAMPLE 3: Forest Validation on Invalid Questions")
    print("-" * 50)
    subject_metadata = load_subject_metadata()
    forest_result = validate_forest_questions(
        result['invalid_questions'], 
        subject_metadata,
        valid_roots=[3, 1642]
    )
    print_forest_validation_report(forest_result, max_display=5)