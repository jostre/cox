import os
import nbformat
from nbconvert import PythonExporter
from pathlib import Path

def convert_notebook_to_python(notebook_path, output_dir=None):
    """
    Convert a single Jupyter notebook to a Python file.
    """
    try:
        # Read the notebook
        with open(notebook_path) as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Configure the exporter
        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(notebook)
        
        # Determine output path
        input_path = Path(notebook_path)
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}.py"
        else:
            output_path = input_path.with_suffix('.py')
            
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the Python file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
            
        return str(output_path)
    
    except Exception as e:
        print(f"Error converting {notebook_path}: {str(e)}")
        return None

def batch_convert_notebooks(input_dir, output_dir=None):
    """
    Convert all Jupyter notebooks in a directory to Python files.
    
    Args:
        input_dir (str): Directory containing .ipynb files
        output_dir (str, optional): Directory to save the output .py files.
                                  If None, saves in the same directory.
    """
    # Get all notebook files
    notebook_files = list(Path(input_dir).glob("**/*.ipynb"))
    
    if not notebook_files:
        print(f"No Jupyter notebooks found in {input_dir}")
        return
    
    print(f"Found {len(notebook_files)} notebook(s)")
    
    # Convert each notebook
    successful = 0
    for notebook_path in notebook_files:
        print(f"Converting {notebook_path}...")
        output_path = convert_notebook_to_python(notebook_path, output_dir)
        
        if output_path:
            successful += 1
            print(f"Successfully converted to {output_path}")
    
    print(f"\nConversion complete: {successful} of {len(notebook_files)} notebooks converted successfully")

if __name__ == "__main__":
    input_directory = "."  # Current directory
    output_directory = "./python_files"  # Output directory
    
    batch_convert_notebooks(input_directory, output_directory)