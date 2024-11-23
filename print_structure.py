import os

def print_structure(root_dir, indent=0):
    """
    Recursively prints the directory structure.
    """
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        print("  " * indent + f"|- {item}")
        if os.path.isdir(item_path):
            print_structure(item_path, indent + 1)

if __name__ == "__main__":
    project_root = "."
    print(f"Project Structure: {os.path.abspath(project_root)}")
    print_structure(project_root)