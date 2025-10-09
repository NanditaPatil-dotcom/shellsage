from __future__ import annotations

import textwrap
import re
from typing import List
import typer
from rich.console import Console
from rich.panel import Panel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from threading import Thread

from utils import retrieve_context, build_or_load_index

# Typer CLI app
app = typer.Typer(add_completion=False, help="ShellSage: Linux command explainer with RAG (FAISS + Hugging Face)")
console = Console()
_generator = None  # lazy-init model
_tokenizer = None

# Max prompt tokens for prompt context
MAX_PROMPT_TOKENS = 1024
RESERVED_NEW_TOKENS = 256  # space for generated text


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


def _truncate_prompt(prompt: str, tokenizer) -> str:
    """Truncate prompt to fit within the model token limit."""
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS - RESERVED_NEW_TOKENS).input_ids
    if tokens.size(1) > MAX_PROMPT_TOKENS - RESERVED_NEW_TOKENS:
        tokens = tokens[:, - (MAX_PROMPT_TOKENS - RESERVED_NEW_TOKENS) :]
        prompt = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return prompt


def _format_context(docs: List[dict], max_chars: int = 300) -> str:
    """Concatenate retrieved docs into a compact string with truncation."""
    parts = []
    for d in docs:
        header = f"[Source: {d.get('path','unknown')}]"
        body = (d.get("text") or "").strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def _show_examples() -> None:
    examples = [
        'python shellsage.py main ls -la',
        'python shellsage.py main ps aux | grep python',
        'python shellsage.py main grep -R "main" ./src',
        'python shellsage.py main find . -type f -name "*.log" -delete',
    ]
    console.print(Panel.fit("\n".join(examples), title="Usage Examples", subtitle="Try these commands"))


def _run(ctx: typer.Context, examples: bool, top_k: int, fast: bool, stream: bool, one_line: bool) -> None:
    """Core execution used by both root and `main` subcommand."""
    if examples:
        _show_examples()
        raise typer.Exit(0)

    # Grab all extra args as the raw command
    raw_command = " ".join(ctx.args)
    norm_cmd = " ".join(raw_command.split()).lower()

    # Heuristic: auto one-line for common simple listings like ls -la / ls -al
    if not one_line:
        if norm_cmd in {"ls -la", "ls -al"} or (norm_cmd.startswith("ls") and "-la" in norm_cmd):
            one_line = True
    if not raw_command:
        console.print("[red] No command provided. Use --examples to see usage.[/red]")
        raise typer.Exit(1)

    # Build/load FAISS index
    build_or_load_index(force_rebuild=False)

    console.rule("[b]ShellSage[/b]")
    console.print(f"[bold cyan]Command:[/bold cyan] {raw_command}")

    # Retrieve context
    docs = retrieve_context(raw_command, top_k=top_k)
    context_str = _format_context(docs) if docs else ""
    if not context_str:
        console.print("[yellow]No docs found in ./docs. Proceeding without extra context.[/yellow]")

    # Prepare prompt for an instruction-following causal LM
    style_rule = "Return a single concise sentence without newlines, lists, or code blocks." if one_line else "Be concise and organized."
    prompt = textwrap.dedent(f"""
        System: You are ShellSage. Explain a Linux command for a beginner. {style_rule} Include: what it does, option breakdown, a practical example, and caveats. If context is available, use it to improve accuracy.

        Context:
        {context_str}

        User: Explain and do not repeat the command verbatim: {raw_command}
    """).strip()

    # Load model and tokenizer
    generator, tokenizer = _get_generator()
    prompt = _truncate_prompt(prompt, tokenizer)

    # Generation arguments
    eos_id = tokenizer.eos_token_id
    # One-line responses are very short; fast also caps tokens
    if one_line:
        max_new = 40
    else:
        max_new = 120 if fast else RESERVED_NEW_TOKENS
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        top_k=20,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        return_full_text=False,
    )

    # For one-line responses, avoid streaming so we can post-process
    if stream and not one_line:
        console.print(Panel("", title="Explanation"))
        # Use a streamer to yield tokens as they are generated
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=generator, args=(prompt,), kwargs={"streamer": streamer, **gen_kwargs})
        thread.start()
        pieces = []
        for token in streamer:
            pieces.append(token)
            console.print(token, end="")
        thread.join()
        console.print()
        console.print("\n[dim]Tip: rerun with --examples to see example usage.[/dim]")
    else:
        out = generator(prompt, **gen_kwargs)
        explanation = out[0]["generated_text"].strip()
        if one_line:
            # Collapse whitespace and trim to first sentence for a crisp one-liner
            collapsed = " ".join(explanation.split())
            m = re.search(r"([^.?!]*[.?!])", collapsed)
            explanation = m.group(1) if m else collapsed
        console.print(Panel(explanation, title="Explanation"))
        console.print("\n[dim]Tip: rerun with --examples to see example usage.[/dim]")


# Single command: use `main` subcommand to run explanations.
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    examples: bool = typer.Option(False, "--examples", help="Show usage examples and exit"),
    top_k: int = typer.Option(1, "--top-k", help="Number of documents to retrieve for context"),
    fast: bool = typer.Option(True, "--fast/--no-fast", help="Faster generation with fewer tokens and smaller context (default: fast)"),
    stream: bool = typer.Option(False, "--stream", help="Stream tokens as they are generated for lower perceived latency"),
    one_line: bool = typer.Option(False, "--one-line", help="Force a single-sentence response"),
):
    _run(ctx, examples, top_k, fast, stream, one_line)


@app.command("rebuild-index")
def rebuild_index():
    """
    Force-rebuild the FAISS index from the docs folder.
    """
    console.rule("[b]Rebuilding FAISS Index[/b]")
    try:
        build_or_load_index(force_rebuild=True)
        console.print("[green] Successfully rebuilt the FAISS index from docs/.[/green]")
    except Exception as e:
        console.print(f"[red] Failed to rebuild index: {e}[/red]")


if __name__ == "__main__":
    app()

