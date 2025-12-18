"""Entry point for the Crag CLI testing."""

import asyncio
from pathlib import Path
from typing import Annotated, List

import typer

from crag import graph
from crag.ingestion import ingest_batch

app = typer.Typer()


@app.command()
def invoke(
    question: Annotated[
        str, typer.Argument(..., help="The question to ask the agent.")
    ],
):
    """Invoke the agent with a question."""
    response = graph.invoke({"question": question})  # type: ignore
    typer.secho(response["generation"], fg=typer.colors.CYAN)


@app.command()
def visualize(
    path: Annotated[
        Path, typer.Argument(..., help="The path to the graph image file.")
    ],
):
    """Visualize the agent's graph and save it as an image."""
    with path.open("wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    typer.secho(f"Graph image saved to {path}", fg=typer.colors.GREEN)


@app.command()
def ingest(
    paths: Annotated[
        List[Path],
        typer.Argument(
            ..., help="A list of paths to documents or directories to ingest."
        ),
    ],
):
    """Ingest documents from the given paths into the vector store."""

    asyncio.run(ingest_batch(paths))
    typer.secho(
        "Ingested all documents into vector store.",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
