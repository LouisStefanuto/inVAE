import typer

app = typer.Typer()


@app.command()
def main():
    print("hello world!")


@app.command()
def bye():
    print("Bye")


if __name__ == "__main__":
    app()
