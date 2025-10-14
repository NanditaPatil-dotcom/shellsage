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
    """Lazy-load the instruction-tuned causal LM (Qwen2.5-0.5B-Instruct, CPU)."""
    global _generator, _tokenizer
    if _generator is None or _tokenizer is None:
        console.print("[yellow]Loading Qwen/Qwen2.5-0.5B-Instruct on CPU. This may take a moment...[/yellow]")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=_tokenizer,
            device=-1,
        )
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


@app.callback(invoke_without_command=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    top_k: int = typer.Option(3, "--top-k", help="Number of docs to retrieve for context"),
):
    """
    Run a Linux command and explain it using ShellSage AI.
    Usage: shellsage "ls -la"
    """
    raw_command = " ".join(ctx.args)
    if not raw_command:
        console.print("[red]❌ No command provided.[/red]")
        raise typer.Exit(1)

    console.rule("[b]ShellSage[/b]")
    console.print(f"[bold cyan]Running Command:[/bold cyan] {raw_command}\n")

    # Run the command and print output
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

    # Simplified + clean prompt
    prompt = textwrap.dedent(f"""
        Explain this Linux command in simple, clear English.
        Include:
        - What the command does
        - Meaning of each option
        - Example output
        - Practical notes (if any)

        Context:
        {context_str}

        Command: {raw_command}
    """).strip()

    # Generate explanation
    try:
        generator, _ = _get_generator()
        out = generator(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )

        # Strip the echoed prompt if model repeats it
        full_text = out[0]["generated_text"]
        explanation = full_text[len(prompt):].strip()

        console.print(Panel(explanation, title="ShellSage Explanation"))
    except Exception as e:
        console.print(f"[red]Failed to generate AI explanation: {e}[/red]")
        console.print("[yellow]Make sure you have internet connection for first-time model download.[/yellow]")


if __name__ == "__main__":
    app()
