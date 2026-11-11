"""Rich + Prompt_Toolkit interactive terminal UI for Custom Claude Code (ccode)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from custom_claude import __version__
from custom_claude.agent_loop import AgentLoop
from custom_claude.context import ProjectContext
from custom_claude.model_engine import ChunkType, ModelEngine, StreamChunk
from custom_claude.tools import ApprovalGate, global_gate, set_yolo_mode

console = Console()

SLASH_COMMANDS = [
    "/init",
    "/test",
    "/model",
    "/compact",
    "/yolo",
    "/break-and-rebuild",
    "/reverse-engineer",
    "/help",
    "/clear",
    "/exit",
    "/quit",
]


def print_banner(model_name: str, yolo: bool) -> None:
    """Print Claude Code style terminal greeting."""
    banner_text = Text()
    banner_text.append("╭────────────────────────────────────────────────────────────╮\n", style="bold cyan")
    banner_text.append("│              CUSTOM CLAUDE CODE (My-Opus)                  │\n", style="bold white")
    banner_text.append("│        Terminal-Native Autonomous Coding Assistant         │\n", style="dim cyan")
    banner_text.append("╰────────────────────────────────────────────────────────────╯", style="bold cyan")

    console.print(banner_text)
    mode_str = "[bold red]YOLO (Auto-Approve)[/bold red]" if yolo else "[bold green]Human-in-the-Loop Safe[/bold green]"
    console.print(f"[dim]Version:[/dim] {__version__}  |  [dim]Backend:[/dim] [cyan]{model_name}[/cyan]  |  [dim]Gate:[/dim] {mode_str}")
    console.print("[dim]Type your request or use slash commands like [/dim][bold cyan]/init[/bold cyan], [bold cyan]/test[/bold cyan], [bold cyan]/help[/bold cyan], [bold cyan]/exit[/bold cyan].\n")


def display_help() -> None:
    """Show interactive command reference."""
    table = Table(title="Custom Claude Code Slash Commands", border_style="cyan")
    table.add_column("Command", style="bold cyan", width=22)
    table.add_column("Description", style="white")

    table.add_row("/init", "Scan repository and generate OPUS.md project context file")
    table.add_row("/test [target]", "Autonomous loop: run pytest, analyze tracebacks, auto-fix code")
    table.add_row("/model <path_or_url>", "Switch model weights file (.pt) or HTTP endpoint on the fly")
    table.add_row("/break-and-rebuild", "Run red-teaming breaker, log flaws, retrain & hot-reload")
    table.add_row("/reverse-engineer [repo]", "Scan repo, synthesize hidden thinking traces, & adapt RecurrentOpus")
    table.add_row("/compact", "Summarize conversation history to free context window tokens")
    table.add_row("/yolo", "Toggle auto-approval mode for all tool calls and bash commands")
    table.add_row("/clear", "Clear the terminal screen")
    table.add_row("/help", "Show this reference table")
    table.add_row("/exit, /quit", "Exit the Custom Claude terminal")

    console.print(table)


class TerminalUI:
    """Manages the interactive terminal session and live streaming output."""

    def __init__(
        self,
        model_target: str = "http://localhost:8080/v1",
        yolo: bool = False,
        max_iterations: int = 10,
    ) -> None:
        self.yolo = yolo
        self.gate = ApprovalGate(yolo=yolo)
        # Synchronize global gate with session gate
        set_yolo_mode(yolo)

        self.engine = ModelEngine(endpoint_or_path=model_target)
        self.context = ProjectContext()
        self.agent = AgentLoop(
            model_engine=self.engine,
            context=self.context,
            gate=self.gate,
            max_iterations=max_iterations,
        )

        completer = WordCompleter(SLASH_COMMANDS, ignore_case=True)
        pt_style = PtStyle.from_dict({"prompt": "ansicyan bold"})
        try:
            self.session: PromptSession = PromptSession(
                history=InMemoryHistory(),
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                style=pt_style,
            )
        except Exception:
            # Fallback for headless / pytest / captured stdout environments
            from prompt_toolkit.output import DummyOutput
            self.session = PromptSession(
                history=InMemoryHistory(),
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                style=pt_style,
                output=DummyOutput(),
            )

    def handle_slash_command(self, user_input: str) -> bool:
        """Process slash commands. Returns True if handled, False otherwise."""
        parts = user_input.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            console.print("[bold yellow]Goodbye! Happy coding.[/bold yellow]")
            sys.exit(0)

        elif cmd == "/clear":
            console.clear()
            print_banner(self.engine.endpoint_or_path, self.gate.yolo)
            return True

        elif cmd == "/help":
            display_help()
            return True

        elif cmd == "/init":
            console.print("[dim]Scanning repository and generating OPUS.md...[/dim]")
            res = self.context.initialize_opus_md(force=True)
            console.print(Panel(res, title="[bold green]Project Memory[/bold green]", border_style="green"))
            return True

        elif cmd == "/yolo":
            self.yolo = not self.yolo
            self.gate.set_yolo(self.yolo)
            set_yolo_mode(self.yolo)
            state_text = "[bold red]ENABLED[/bold red]" if self.yolo else "[bold green]DISABLED (Safe)[/bold green]"
            console.print(f"YOLO Mode is now {state_text}.")
            return True

        elif cmd == "/compact":
            msg = self.agent.compact_history()
            console.print(f"[bold green]{msg}[/bold green]")
            return True

        elif cmd == "/model":
            if not arg:
                console.print(f"Current model backend: [cyan]{self.engine.endpoint_or_path}[/cyan] ({self.engine.backend})")
                console.print("[dim]Usage: /model <weights_path.pt | http://localhost:8080/v1>[/dim]")
            else:
                notice = self.engine.switch_target(arg)
                console.print(f"[bold green]{notice}[/bold green]")
            return True

        elif cmd == "/test":
            target_file = arg if arg else None
            console.print(f"[bold cyan]Starting Autonomous Test & Repair Loop ({target_file or 'All Tests'})...[/bold cyan]")
            self.stream_agent_response(lambda callback: self.agent.run_autonomous_test_loop(target_file, on_chunk=callback))
            return True

        elif cmd == "/break-and-rebuild":
            console.print("[bold cyan]Initiating Adversarial Breaker & Model Rebuild Pipeline...[/bold cyan]")
            try:
                from rebuild_engine.diagnose_and_synth import FlawDatasetManager
                mgr = FlawDatasetManager(output_path="data/flaw_traces.jsonl")
                records = mgr.generate_benchmark_flaws()
                console.print(f"[dim]Harvested {len(records)} flaw traces across deceptive bug traps and mutants.[/dim]")

                from reverse_eng.recurrent_opus import RecurrentOpusConfig, RecurrentOpusModel
                from rebuild_engine.rebuilder import ModelRebuilder

                cfg = RecurrentOpusConfig(vocab_size=32000, dim=64, n_layers=3, n_heads=4, num_recurrences=3)
                base_model = RecurrentOpusModel(cfg)

                rebuilder = ModelRebuilder(base_model, lr=1e-4)
                ckpt_path = "rebuilt_opus.pt"
                res = rebuilder.train_on_flaws(
                    flaw_jsonl_path="data/flaw_traces.jsonl",
                    epochs=1,
                    batch_size=4,
                    save_path=ckpt_path,
                )
                console.print(f"[bold green]Rebuild complete! Steps: {res['steps']}, Initial Loss: {res['initial_loss']}, Final Loss: {res['final_loss']}[/bold green]")

                reload_msg = self.engine.switch_target(str(Path(ckpt_path).resolve()))
                console.print(f"[bold green]Hot-reloaded into CLI engine: {reload_msg}[/bold green]")
            except Exception as exc:
                console.print(f"[bold red]Error in break-and-rebuild pipeline: {exc}[/bold red]")
            return True

        elif cmd == "/reverse-engineer":
            target_repo = arg if arg else "."
            console.print(f"[bold cyan]Starting Reverse-Engineering & Distillation on `{target_repo}`...[/bold cyan]")
            try:
                from reverse_eng.context_packer import build_repo_skeleton
                skeleton = build_repo_skeleton(target_repo, max_files=20)
                console.print(f"[dim]Generated AST Repo Skeleton ({len(skeleton)} chars).[/dim]")

                from reverse_eng.trace_synthesizer import TraceSynthesizer
                synth = TraceSynthesizer(output_path="data/reverse_engineered_traces.jsonl")
                from rebuild_engine.breaker import DECEPTIVE_BUG_TRAPS
                sample_count = 0
                for trap in DECEPTIVE_BUG_TRAPS[:5]:
                    synth.synthesize_sample(
                        bug_id=trap.name,
                        buggy_code=trap.buggy_code,
                        fixed_code=trap.fixed_code,
                        test_func_name=f"test_{trap.name}",
                        failing_traceback=f"AssertionError in {trap.name}",
                    )
                    sample_count += 1
                console.print(f"[bold green]Synthesized {sample_count} training traces -> data/reverse_engineered_traces.jsonl[/bold green]")

                from reverse_eng.recurrent_opus import RecurrentOpusConfig, RecurrentOpusModel
                cfg = RecurrentOpusConfig(vocab_size=1000, dim=64, n_layers=3, n_heads=4, num_recurrences=3)
                _ = RecurrentOpusModel(cfg)
                console.print(f"[bold green]RecurrentOpusModel initialized with K=3 recurrent thinking depth.[/bold green]")
            except Exception as exc:
                console.print(f"[bold red]Error in reverse-engineering pipeline: {exc}[/bold red]")
            return True

        return False

    def stream_agent_response(self, execution_fn) -> None:
        """Stream model thinking in dim gray and output in standard format."""
        current_phase = None

        def on_chunk(chunk: StreamChunk) -> None:
            nonlocal current_phase
            if chunk.chunk_type == ChunkType.THINKING:
                if current_phase != "thinking":
                    current_phase = "thinking"
                    console.print("\n[dim italic]Thinking:[/dim italic] ", end="")
                console.print(Text(chunk.content, style="dim italic"), end="")
            elif chunk.chunk_type == ChunkType.TEXT:
                if current_phase == "thinking":
                    current_phase = "text"
                    console.print("\n")
                console.print(chunk.content, end="")
            elif chunk.chunk_type == ChunkType.TOOL_CALL and chunk.tool_call:
                console.print(
                    f"\n[bold cyan][*] Executing Tool:[/bold cyan] [yellow]{chunk.tool_call.name}[/yellow] "
                    f"[dim]{chunk.tool_call.arguments}[/dim]"
                )
            elif chunk.chunk_type == ChunkType.FINISH:
                console.print()

        try:
            execution_fn(on_chunk)
            console.print()
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user.[/bold red]")
        except Exception as exc:
            console.print(f"\n[bold red]Error running turn:[/bold red] {exc}")

    def run_interactive(self) -> None:
        """Main REPL loop."""
        print_banner(self.engine.endpoint_or_path, self.gate.yolo)

        while True:
            try:
                user_input = self.session.prompt("opus > ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold yellow]Exiting Custom Claude Code.[/bold yellow]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self.handle_slash_command(user_input):
                    continue

            # Standard prompt
            self.stream_agent_response(lambda callback: self.agent.run_turn(user_input, on_chunk=callback))


def main() -> None:
    """CLI entrypoint for ccode command."""
    parser = argparse.ArgumentParser(
        prog="ccode",
        description="Custom Claude Code - Anthropic Claude Code terminal experience powered by My-Opus.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional initial instruction to run non-interactively")
    parser.add_argument("--yolo", action="store_true", help="Auto-approve all tool calls and bash executions")
    parser.add_argument(
        "--model",
        default="http://localhost:8080/v1",
        help="Path to PyTorch weights .pt or local server URL (default: http://localhost:8080/v1)",
    )
    parser.add_argument("--max-iterations", type=int, default=10, help="Max tool execution iterations per turn")
    args = parser.parse_args()

    ui = TerminalUI(model_target=args.model, yolo=args.yolo, max_iterations=args.max_iterations)

    if args.prompt:
        single_prompt = " ".join(args.prompt).strip()
        if single_prompt.startswith("/"):
            if ui.handle_slash_command(single_prompt):
                return
        ui.stream_agent_response(lambda callback: ui.agent.run_turn(single_prompt, on_chunk=callback))
    else:
        ui.run_interactive()


if __name__ == "__main__":
    main()
