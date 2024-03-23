from typing import Annotated
from typing import Optional
import typer
import random


# basic
# def main(name: str, lastname: str = 'ds', formal: bool = True):
#     if formal:
#         print(f"Good day Ms. {name} {lastname}.")
#     else:
#         print(f"Hello {name} {lastname}")


# abort, exit code is 1.
# def main(username: str):
#     if username == "root":
#         print("The root user is reserved")
#         raise typer.Abort()
#     print(f"New user created: {username}")


# https://typer.tiangolo.com/tutorial/arguments/optional/
def main(name: Annotated[Optional[str], typer.Argument()] = None):
    if name is None:
        print("Hello World!")
    else:
        print(f"Hello {name}")


# def get_name():
#     return random.choice(["Deadpool", "Rick", "Morty", "Hiro"])


# def main(name: Annotated[str, typer.Argument(default_factory=get_name)]):
#     print(f"Hello {name}")


# def main(name: Annotated[str, typer.Argument(envvar="AWESOME_NAME")] = "World"):
#     print(f"Hello Mr. {name}")


# def main(
#     name: Annotated[str, typer.Argument(envvar=["AWESOME_NAME", "GOD_NAME"])] = "World"
# ):
#     print(f"Hello Mr. {name}")


if __name__ == "__main__":
    typer.run(main)
