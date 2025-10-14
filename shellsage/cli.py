from __future__ import annotations
import subprocess
import textwrap
import typer
from rich.console import Console
from rich.panel import Panel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from shellsage.utils import retrieve_context, build_or_load_index

app = typer.Typer(
    add_completion=False,
    help="ShellSage: Explain Linux commands using docs + AI"
)
console = Console()
_generator = None
_tokenizer = None


def _get_generator():
    """Lazy-load the instruction-tuned LM (Qwen2.5-0.5B-Instruct, CPU)."""
    global _generator, _tokenizer
    if _generator is None or _tokenizer is None:
        console.print("[yellow]Loading Qwen2.5-0.5B-Instruct on CPU...[/yellow]")
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


def _format_context(docs, max_chars: int = 500) -> str:
    parts = []
    for d in docs:
        header = f"[Source: {d.get('path', 'unknown')}]"
        body = (d.get("text") or "").strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _clean_explanation(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Find the first sentence ending with ., !, or ?
    for i, char in enumerate(text):
        if char in '.!?':
            return text[:i+1].strip()
    # If no punctuation found, take the first line and add a period
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip()
        if not first_line.endswith('.'):
            first_line += '.'
        return first_line
    return ""


@app.callback(invoke_without_command=True, context_settings={"ignore_unknown_options": True})
def main(
    raw_command: str = typer.Argument(..., help='The Linux command to explain, e.g., "ls -la"'),
    top_k: int = typer.Option(3, "--top-k", help="Number of docs to retrieve for context"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force rebuild of the FAISS index"),
):
    console.rule("[b]ShellSage[/b]")
    console.print(f"[bold cyan]Running Command:[/bold cyan] {raw_command}\n")

    # Rebuild FAISS index if requested or missing
    if rebuild:
        console.print("[yellow]Rebuilding FAISS index...[/yellow]")
        build_or_load_index(force_rebuild=True)
    else:
        build_or_load_index(force_rebuild=False)

    # Run the Linux command
    try:
        result = subprocess.run(raw_command, shell=True, check=False, text=True, capture_output=True)
        console.print(result.stdout or "[dim]No output[/dim]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to run command: {e}[/red]")

    # Retrieve context from docs
    docs = retrieve_context(raw_command, top_k=top_k)
    context_str = _format_context(docs) if docs else ""

    prompt = textwrap.dedent(f"""
        Explain in one sentence what this Linux command does.
        Context:
        {context_str}
        Command:
        {raw_command}
        Explanation:
    """).strip()

    try:
        generator, _ = _get_generator()
        out = generator(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )

        full_text = out[0]["generated_text"]
        explanation = full_text[len(prompt):].strip()
        explanation = _clean_explanation(explanation)

        if not explanation:
            first_word = raw_command.split()[0]
            explanation = f"No explanation found in docs. But '{raw_command}' is a Linux command — try checking 'man {first_word}' for details."

        console.print(Panel(explanation, title="ShellSage Explanation"))
    except Exception as e:
        console.print(f"[red]Failed to generate AI explanation: {e}[/red]")
        console.print("[yellow]Ensure internet connection for first-time model download.[/yellow]")


if __name__ == "__main__":
    app()

