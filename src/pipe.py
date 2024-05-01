# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

import math
import numpy as np

from sys import stdin
from typing import Type
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

piece_to_code = {
    "FC": 0,
    "FB": 1,
    "FE": 2,
    "FD": 3,
    "BC": 4,
    "BB": 5,
    "BE": 6,
    "BD": 7,
    "VC": 8,
    "VB": 9,
    "VE": 10,
    "VD": 11,
    "LH": 12,
    "LV": 13
}

code_to_piece = (
    "FC",
    "FB",
    "FE",
    "FD",
    "BC",
    "BB",
    "BE",
    "BD",
    "VC",
    "VB",
    "VE",
    "VD",
    "LH",
    "LV"
)


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, size: int, storage: np.array) -> None:
        self.size: int = size
        self.storage: np.array = storage

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return code_to_piece(self.storage[row * self.size + col])

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (
            code_to_piece[self.storage[(row - 1) * self.size + col]] if row > 0 else None,
            code_to_piece[self.storage[(row + 1) * self.size + col]] if row < self.size - 1 else None
        )

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (
            code_to_piece[self.storage[row * self.size + col - 1]] if col > 0 else None,
            code_to_piece[self.storage[row * self.size + col + 1]] if col < self.size - 1 else None
        )

    def print(self) -> str:
        i = 0
        out = ""

        # Loops through storage and builds the output
        for piece_code in self.storage.tolist():
            i += 1
            out += code_to_piece[piece_code]
            if i % self.size == 0:
                if i == self.size ** 2:
                    return out
                out += '\n'
            else:
                out += '\t'
        return out

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        lines = stdin.readlines()

        # Stores every piece of the input in a np.array
        storage = np.array('\t'.join(lines).replace('\n', '').split('\t'))

        # Converts the visual representation of all pieces into its interal representation
        storage = np.vectorize(Board.convert_piece)(storage)

        return Board(int(math.sqrt(len(storage))), storage)

    def convert_piece(piece: str) -> int:
        return piece_to_code[piece]

    # TODO: outros metodos da classe


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
