import pandas as pd
from typing import Set, Dict, List, Tuple
from collections import defaultdict


def validate_forest_structure(csv_path: str) -> Dict:
    """
    Validates if the subject metadata forms a forest (collection of trees).
    
    A valid forest must have:
    1. One or more root nodes (nodes with no parent)
    2. Every non-root node has exactly one parent that exists in the dataset
    3. No cycles
    4. All nodes belong to some tree (all nodes are connected to a root)
    
    Args:
        csv_path: Path to the subject_metadata.csv file
        
    Returns:
        Dictionary containing forest validation results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Replace 'NULL' strings with None
    df['ParentId'] = df['ParentId'].replace('NULL', None)
    
    print(f"Total nodes: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Extract data structures
    all_subject_ids = set(df['SubjectId'].values)
    parent_to_children: Dict[int, List[int]] = defaultdict(list)
    subject_to_parent: Dict[int, int] = {}
    subject_to_name: Dict[int, str] = {}
    root_nodes: List[int] = []
    
    # Build parent-child relationships and name mappings
    for _, row in df.iterrows():
        subject_id = row['SubjectId']
        parent_id = row['ParentId']
        name = row['Name']
        
        subject_to_name[subject_id] = name
        
        if pd.isna(parent_id):
            root_nodes.append(subject_id)
        else:
            parent_id = int(parent_id)
            subject_to_parent[subject_id] = parent_id
            parent_to_children[parent_id].append(subject_id)
    
    results = {
        'is_valid_forest': True,
        'is_single_tree': len(root_nodes) == 1,
        'issues': [],
        'root_nodes': root_nodes,
        'total_nodes': len(df),
        'num_roots': len(root_nodes),
        'trees': []
    }
    
    # Check 1: At least one root node
    if len(root_nodes) == 0:
        results['is_valid_forest'] = False
        results['issues'].append("❌ No root nodes found (no nodes with NULL parent)")
        return results
    else:
        results['issues'].append(f"✓ Found {len(root_nodes)} root node(s)")
    
    # Check 2: All parent IDs exist in the dataset
    orphaned_nodes = []
    for subject_id in all_subject_ids:
        if subject_id not in root_nodes and subject_id in subject_to_parent:
            parent = subject_to_parent[subject_id]
            if parent not in all_subject_ids:
                orphaned_nodes.append((subject_id, parent))
    
    if orphaned_nodes:
        results['is_valid_forest'] = False
        results['issues'].append(f"❌ {len(orphaned_nodes)} nodes have non-existent parents:")
        for subject_id, missing_parent in orphaned_nodes:
            results['issues'].append(f"   - SubjectId {subject_id} references parent {missing_parent} which doesn't exist")
    else:
        results['issues'].append("✓ All nodes have valid parents (all parent IDs exist in dataset)")
    
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
    
    for node in all_subject_ids:
        if node not in visited:
            if has_cycle_from_node(node, visited, set()):
                has_cycle = True
                break
    
    if has_cycle:
        results['is_valid_forest'] = False
        results['issues'].append("❌ Cycle detected in the forest structure")
    else:
        results['issues'].append("✓ No cycles detected")
    
    # Check 4: All nodes are reachable from some root (all nodes belong to some tree)
    def get_tree_nodes_and_stats(root: int) -> Tuple[Set[int], int, int]:
        """
        Returns (reachable_nodes, max_depth, node_count) for a tree rooted at root
        """
        reachable = set()
        queue = [(root, 0)]  # (node, depth)
        reachable.add(root)
        max_depth = 0
        
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            
            if node in parent_to_children:
                for child in parent_to_children[node]:
                    if child not in reachable:
                        reachable.add(child)
                        queue.append((child, depth + 1))
        
        return reachable, max_depth, len(reachable)
    
    # Analyze each tree in the forest
    total_reachable = set()
    for root in sorted(root_nodes):
        reachable, max_depth, node_count = get_tree_nodes_and_stats(root)
        total_reachable.update(reachable)
        
        tree_info = {
            'root_id': root,
            'root_name': subject_to_name[root],
            'nodes_in_tree': node_count,
            'max_depth': max_depth,
            'node_ids': sorted(reachable)
        }
        results['trees'].append(tree_info)
    
    unreachable = all_subject_ids - total_reachable
    if unreachable:
        results['is_valid_forest'] = False
        results['issues'].append(f"❌ {len(unreachable)} nodes are unreachable from any root: {unreachable}")
    else:
        results['issues'].append(f"✓ All nodes are reachable from some root")
    
    return results


def print_forest_report(results: Dict):
    """Print a formatted forest validation report"""
    print("="*70)
    print("FOREST STRUCTURE VALIDATION REPORT")
    print("="*70)
    print()
    
    print("Validation Results:")
    print("-" * 70)
    for issue in results['issues']:
        print(issue)
    
    print()
    print("="*70)
    print("FOREST SUMMARY")
    print("="*70)
    if results['is_valid_forest']:
        print("✓ VALID FOREST STRUCTURE")
    else:
        print("✗ INVALID FOREST STRUCTURE")
    
    print(f"\nTotal nodes: {results['total_nodes']}")
    print(f"Number of trees in forest: {results['num_roots']}")
    
    if results['is_single_tree']:
        print("Note: This is a SINGLE TREE, not a forest")
    else:
        print(f"Note: This is a FOREST with {results['num_roots']} separate trees")
    
    # Print details for each tree
    print()
    print("="*70)
    print("TREE DETAILS")
    print("="*70)
    for idx, tree in enumerate(results['trees'], 1):
        print(f"\nTree {idx}:")
        print(f"  Root ID: {tree['root_id']}")
        print(f"  Root Name: {tree['root_name']}")
        print(f"  Number of nodes: {tree['nodes_in_tree']}")
        print(f"  Maximum depth: {tree['max_depth']}")
        print(f"  Node IDs: {tree['node_ids'][:10]}", end="")
        if len(tree['node_ids']) > 10:
            print(f" ... and {len(tree['node_ids']) - 10} more")
        else:
            print()


def main():
    csv_path = '/Users/atlasbuchholz/PycharmProjects/UBC/STAT_405_Project/data/metadata/subject_metadata.csv'
    
    results = validate_forest_structure(csv_path)
    print_forest_report(results)
    
    # Summary
    print()
    print("="*70)
    print("ROOTS OF THE FOREST")
    print("="*70)
    for root_id, tree in zip(results['root_nodes'], results['trees']):
        print(f"\nRoot {root_id}: {tree['root_name']}")
        print(f"  Contains {tree['nodes_in_tree']} nodes with max depth {tree['max_depth']}")


if __name__ == "__main__":
    main()
