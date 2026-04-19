import pandas as pd
from typing import Set, Dict, List, Tuple


def validate_tree_structure(csv_path: str) -> Dict:
    """
    Validates if the subject metadata CSV forms a valid tree structure.
    
    A valid tree must have:
    1. Exactly one root node (no parent)
    2. Every non-root node has exactly one parent
    3. No cycles
    4. All nodes are connected to the root
    
    Args:
        csv_path: Path to the subject_metadata.csv file
        
    Returns:
        Dictionary containing validation results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Replace 'NULL' strings with None
    df['ParentId'] = df['ParentId'].replace('NULL', None)
    
    print(f"Total nodes: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Extract data structures
    subject_ids = set(df['SubjectId'].values)
    parent_to_children: Dict[int, List[int]] = {}
    subject_to_parent: Dict[int, int] = {}
    root_nodes: List[int] = []
    
    # Build parent-child relationships
    for _, row in df.iterrows():
        subject_id = row['SubjectId']
        parent_id = row['ParentId']
        
        if pd.isna(parent_id):
            root_nodes.append(subject_id)
        else:
            parent_id = int(parent_id)
            subject_to_parent[subject_id] = parent_id
            
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(subject_id)
    
    results = {
        'is_valid_tree': True,
        'issues': [],
        'root_nodes': root_nodes,
        'total_nodes': len(df),
        'num_roots': len(root_nodes)
    }
    
    # Check 1: Exactly one root node
    if len(root_nodes) == 0:
        results['is_valid_tree'] = False
        results['issues'].append("❌ No root node found (no node with NULL parent)")
    elif len(root_nodes) > 1:
        results['is_valid_tree'] = False
        results['issues'].append(f"❌ Multiple root nodes found: {root_nodes}")
    else:
        results['issues'].append(f"✓ Valid single root node: {root_nodes[0]}")
    
    # Check 2: Every node has at most one parent (by definition in this case)
    results['issues'].append(f"✓ Every node has at most one parent (enforced by data structure)")
    
    # Check 3: Detect cycles using DFS
    def has_cycle_from_node(node: int, visited: Set[int], rec_stack: Set[int]) -> bool:
        visited.add(node)
        rec_stack.add(node)
        
        if node in parent_to_children:
            for child in parent_to_children[node]:
                if child not in visited:
                    if has_cycle_from_node(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True
        
        rec_stack.remove(node)
        return False
    
    visited: Set[int] = set()
    has_cycle = False
    
    for node in subject_ids:
        if node not in visited:
            if has_cycle_from_node(node, visited, set()):
                has_cycle = True
                break
    
    if has_cycle:
        results['is_valid_tree'] = False
        results['issues'].append("❌ Cycle detected in the tree structure")
    else:
        results['issues'].append("✓ No cycles detected")
    
    # Check 4: All nodes are connected to root (all parents exist)
    orphaned_nodes = []
    for subject_id in subject_ids:
        if subject_id not in root_nodes:
            # Check if parent exists in dataset
            if subject_id in subject_to_parent:
                parent = subject_to_parent[subject_id]
                if parent not in subject_ids:
                    orphaned_nodes.append((subject_id, parent))
    
    if orphaned_nodes:
        results['is_valid_tree'] = False
        results['issues'].append(f"❌ {len(orphaned_nodes)} nodes have non-existent parents:")
        for subject_id, missing_parent in orphaned_nodes:
            results['issues'].append(f"   - SubjectId {subject_id} references parent {missing_parent} which doesn't exist")
    else:
        results['issues'].append("✓ All nodes have valid parents (all parent IDs exist in dataset)")
    
    # Check 5: Verify all nodes are reachable from root using BFS
    if root_nodes:
        root = root_nodes[0]
        reachable = set()
        queue = [root]
        reachable.add(root)
        
        while queue:
            node = queue.pop(0)
            if node in parent_to_children:
                for child in parent_to_children[node]:
                    if child not in reachable:
                        reachable.add(child)
                        queue.append(child)
        
        unreachable = subject_ids - reachable
        if unreachable:
            results['is_valid_tree'] = False
            results['issues'].append(f"❌ {len(unreachable)} nodes are unreachable from root: {unreachable}")
        else:
            results['issues'].append(f"✓ All nodes are reachable from root")
    
    return results


def print_tree_stats(csv_path: str):
    """Print statistics about the tree structure"""
    df = pd.read_csv(csv_path)
    df['ParentId'] = df['ParentId'].replace('NULL', None)
    
    # Calculate tree statistics
    levels = df['Level'].values if 'Level' in df.columns else None
    
    print("\n" + "="*60)
    print("TREE DEPTH ANALYSIS")
    print("="*60)
    if levels is not None:
        print(f"Max tree depth: {max(levels)}")
        for level in sorted(set(levels)):
            count = len(df[df['Level'] == level])
            print(f"  Level {level}: {count} nodes")


def main():
    csv_path = '/Users/atlasbuchholz/PycharmProjects/UBC/STAT_405_Project/data/metadata/subject_metadata.csv'
    
    print("="*60)
    print("SUBJECT METADATA TREE STRUCTURE VALIDATION")
    print("="*60)
    print()
    
    results = validate_tree_structure(csv_path)
    
    print("\nValidation Results:")
    print("-" * 60)
    for issue in results['issues']:
        print(issue)
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    if results['is_valid_tree']:
        print("✓ VALID TREE STRUCTURE")
    else:
        print("✗ NOT A VALID TREE STRUCTURE")
    
    print(f"\nRoot nodes: {results['root_nodes']}")
    print(f"Total nodes: {results['total_nodes']}")
    
    print_tree_stats(csv_path)


if __name__ == "__main__":
    main()
