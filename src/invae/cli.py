from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import typer
from typing_extensions import Annotated

app = typer.Typer(no_args_is_help=True)


def get_version() -> str:
    try:
        return version("invae")
    except PackageNotFoundError:
        return "0.0.0 (local)"


def version_callback(value: bool) -> None:
    if value:
        print(f"inVAE Version: {get_version()}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback),
    ] = None,
) -> None:
    pass


if __name__ == "__main__":
    app()
