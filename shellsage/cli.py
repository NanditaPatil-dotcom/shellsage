from __future__ import annotations
import subprocess
import textwrap
from typing import List
import typer
from rich.console import Console
from rich.panel import Panel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .utils import retrieve_context, build_or_load_index

app = typer.Typer(add_completion=False, help="ShellSage: Linux command explainer with RAG (FAISS + Hugging Face)")
console = Console()
_generator = None
_tokenizer = None


def _get_generator():
    """Lazy-load the instruction-tuned causal LM."""
    global _generator, _tokenizer
    if _generator is None or _tokenizer is None:
        console.print("[yellow]Loading Qwen/Qwen2.5-0.5B-Instruct on CPU. This may take a moment...[/yellow]")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _generator = pipeline("text-generation", model=model, tokenizer=_tokenizer, device=-1)
    return _generator, _tokenizer


def _format_context(docs: List[dict], max_chars: int = 500) -> str:
    parts = []
    for d in docs:
        header = f"[Source: {d.get('path', 'unknown')}]"
        body = (d.get("text") or "").strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


@app.command()
def main(
    command: List[str] = typer.Argument(..., help="Linux command to explain (in quotes, e.g. 'ls -la')"),
    top_k: int = typer.Option(3, "--top-k", help="Number of docs to retrieve for context"),
):
    """
    Run a Linux command and explain it using ShellSage AI.
    Example:
        shellsage main "ls -la"
    """
    raw_command = " ".join(command)
    console.rule("[b]ShellSage[/b]")
    console.print(f"[bold cyan]Running:[/bold cyan] {raw_command}\n")

    # Run command
    try:
        result = subprocess.run(raw_command, shell=True, check=False, text=True, capture_output=True)
        console.print(result.stdout or "[dim]No output[/dim]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to run command: {e}[/red]")

    # Load FAISS index
    build_or_load_index(force_rebuild=False)

    # Retrieve context
    docs = retrieve_context(raw_command, top_k=top_k)
    context_str = _format_context(docs) if docs else ""

    # Build AI prompt
    prompt = textwrap.dedent(f"""
    You are ShellSage — a friendly AI that explains Linux commands in clear, beginner-friendly English.
    Explain what the command does, the meaning of each option, and give a short example.

    Context:
    {context_str}

    Command:
    {raw_command}

    Explanation:
    """).strip()

    # Generate explanation
    try:
        generator, _ = _get_generator()
        out = generator(prompt, max_new_tokens=256, num_beams=4)
        explanation = out[0]["generated_text"].strip()
        console.print(Panel(explanation, title="ShellSage Explanation", border_style="bright_blue"))
    except Exception as e:
        console.print(f"[red]Failed to generate explanation: {e}[/red]")
        console.print("[yellow]Make sure you have internet for first-time model download.[/yellow]")


if __name__ == "__main__":
    app()


