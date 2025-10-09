# ShellSage

ShellSage is a user-ready CLI that explains Linux commands in plain English using Retrieval-Augmented Generation (RAG). It retrieves context from local docs via FAISS and generates explanations with a Hugging Face model.

## Project Structure

\`\`\`
ShellSage/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── shellsage.py              # CLI entrypoint (Typer)
├── utils.py                  # RAG utilities: embed, index, retrieve
├── docs/                     # Knowledge base (embedded for retrieval)
│   ├── ls.txt
│   ├── ps.txt
│   └── grep.txt
├── embeddings/               # Saved FAISS index + metadata
│   └── .gitkeep
└── models/                   # Optional local HF models (empty by default)
    └── .gitkeep
\`\`\`

- docs/: Put text files here to expand ShellSage’s knowledge base.
- embeddings/: Contains `shell_docs.index` (FAISS) and `shell_docs_meta.json` (doc mapping) after indexing.
- models/: If you want to store downloaded/local HF models manually (optional).

## Installation

1) Ensure Python 3.9+ and internet access are available.
2) Install dependencies:
\`\`\`
pip install -r requirements.txt
\`\`\`

## Indexing

The first run of the CLI will build the FAISS index automatically from files in `docs/`.  
You can also rebuild manually:
\`\`\`
python utils.py
\`\`\`

## Usage

Explain a command:
\`\`\`
python shellsage.py "ls -la"
\`\`\`

Show usage examples:
\`\`\`
python shellsage.py --examples
\`\`\`

Options:
- `--top-k` Number of docs to retrieve (default: 3).

## Models and Internet

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generation: `google/flan-t5-small` via `transformers.pipeline("text2text-generation")`

These models are downloaded on first use and require an internet connection.

## Create a Standalone Executable (PyInstaller)

Install PyInstaller:
\`\`\`
pip install pyinstaller
\`\`\`

Build (Linux/macOS):
\`\`\`
pyinstaller --onefile --name shellsage shellsage.py
\`\`\`

Build (Windows):
\`\`\`
pyinstaller --onefile --name shellsage.exe shellsage.py
\`\`\`

Run the executable (after build):
- Linux/macOS:
  \`\`\`
  ./dist/shellsage "ls -la"
  \`\`\`
- Windows:
  \`\`\`
  .\dist\shellsage.exe "ls -la"
  \`\`\`

Notes:
- The first run may still download models from Hugging Face if not already cached.
- If you need fully offline usage, pre-download models into a local path and adjust the `pipeline` call accordingly.
