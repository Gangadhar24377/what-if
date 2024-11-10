import os

# Define the folder structure
structure = {
    "what-if": [
        "data/__init__.py",
        "data/data_loader.py",
        "models/__init__.py",
        "models/classifier.py",
        "models/counterfactual.py",
        "utils/__init__.py",
        "utils/visualization.py",
        "main.py",
        "requirements.txt"
    ]
}

# Function to create the folder structure
def create_structure(structure):
    for root, files in structure.items():
        for file_path in files:
            # Create directories if they don't exist
            full_path = os.path.join(root, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # Create each file
            with open(full_path, "w") as f:
                pass  # Creates an empty file

# Execute the function
create_structure(structure)
print("Folder structure created successfully.")
