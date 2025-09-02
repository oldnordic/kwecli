import os

def read_markdown_files(directory: str) -> str:
    """Reads all markdown files in a directory and returns their concatenated content."""
    content = ""
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), "r") as f:
                content += f.read() + "\n\n"
    return content