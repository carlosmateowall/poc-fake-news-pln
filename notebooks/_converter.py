"""Converte notebooks em formato percent (.py) para .ipynb. Uso interno, sem dependências."""
import json
import sys
from pathlib import Path


def converter(caminho_py: Path) -> Path:
    linhas = caminho_py.read_text(encoding="utf-8").splitlines()
    celulas = []
    tipo, corpo = None, []

    def fechar():
        if tipo is None:
            return
        texto = "\n".join(corpo).strip("\n")
        if not texto.strip():
            return
        if tipo == "markdown":
            src = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                   for l in texto.splitlines()]
            celulas.append({"cell_type": "markdown", "metadata": {},
                            "source": [l + "\n" for l in src[:-1]] + [src[-1]]})
        else:
            src = texto.splitlines()
            celulas.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                            "outputs": [],
                            "source": [l + "\n" for l in src[:-1]] + [src[-1]]})

    for linha in linhas:
        if linha.startswith("# %% [markdown]"):
            fechar(); tipo, corpo = "markdown", []
        elif linha.startswith("# %%"):
            fechar(); tipo, corpo = "code", []
        elif tipo is not None:
            corpo.append(linha)
    fechar()

    nb = {
        "cells": celulas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destino = caminho_py.with_suffix(".ipynb")
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return destino


if __name__ == "__main__":
    for nome in sys.argv[1:]:
        print(converter(Path(nome)))
