# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

import math
import numpy as np

from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

    

# High nibble stores lock bit (1st bit) and piece type (F -> 00, B -> 01, V -> 10, L -> 11) in the last 2 bits.
# Low nibble stores open ends (0bXXXX_(Left)(Up)(Right)(Down) )
piece_to_byte = {
    "FC": 0x4,
    "FB": 0x1,
    "FE": 0x8,
    "FD": 0x2,
    "BC": 0x1E,
    "BB": 0x1B,
    "BE": 0x1D,
    "BD": 0x17,
    "VC": 0x2C,
    "VB": 0x23,
    "VE": 0x29,
    "VD": 0x26,
    "LH": 0x3A,
    "LV": 0x35
}

byte_to_piece = {
    0x4: "FC",
    0x1: "FB",
    0x8: "FE",
    0x2: "FD",
    0x1E: "BC",
    0x1B: "BB",
    0x1D: "BE",
    0x17: "BD",
    0x2C: "VC",
    0x23: "VB",
    0x29: "VE",
    0x26: "VD",
    0x3A: "LH",
    0x35: "LV"
}

a = {
    0b0011: (0b0010, 0b0001),
    0b0110: (0b0100, 0b0010),
    0b1100: (0b1000, 0b0100),
    0b1001: (0b1000, 0b0001),
    0b1010: (0b1000, 0b0010),
    0b0101: (0b0100, 0b0001),
}


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board: Board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, size: int, storage: list) -> None:
        self.size = size
        self.storage = storage

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return byte_to_piece(self.storage[row * self.size + col])

    def adjacent_vertical_values(self, row: int, col: int, index = -1) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if index != -1:
            return (
                self.storage[index - self.size] if index >= self.size else None,
                self.storage[index + self.size] if (self.size ** 2) - index > self.size else None
            )
        return (
            self.storage[(row - 1) * self.size + col] if row > 0 else None,
            self.storage[(row + 1) * self.size + col] if row < self.size - 1 else None
        )

    def adjacent_horizontal_values(self, row: int, col: int, index = -1) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if index != -1:
            return (
                self.storage[index - 1] if index % self.size != 0 else None,
                self.storage[index + 1] if (index + 1) % self.size != 0 else None
            )
        return (
            self.storage[row * self.size + col - 1] if col > 0 else None,
            self.storage[row * self.size + col + 1] if col < self.size - 1 else None
        )

    def print(self) -> str:
        # Loops through storage and builds the output
        out = []
        for i, piece in enumerate(self.storage):
            out.append(byte_to_piece[piece & 0x7F]) # Remove lock bit
            if ((i + 1) % self.size != 0):
                out.append('\t')
            elif ((i + 1) % self.size == 0) and i != self.size ** 2 - 1:
                out.append('\n')

        return "".join(out)
    
    def copy(self):
        return Board(self.size, self.storage.copy())

    def change_piece(self,index, new_piece):
        if not self.isLocked(index):
            self.storage[index] = new_piece
        
    def __eq__(self, other) -> bool:
        return self.storage == other.storage
    
    def lockPiece(self, index):
        self.storage[index] |= 0x80
        
    def isLocked(self, index):
        return True if self.storage[index] & 0x80 else False
        
    def get_piece_adjs(self, index, piece):
        h_adjs = self.adjacent_horizontal_values(-1, -1, index)
        v_adjs = self.adjacent_vertical_values(-1, -1, index)
        
        adjs = []
        if piece & 0b1000:
            adjs.append(h_adjs[0])
        if piece & 0b0100:
            adjs.append(v_adjs[0])
        if piece & 0b0010:
            adjs.append(h_adjs[1])
        if piece & 0b0001:
            adjs.append(v_adjs[1])
        
        return adjs
    
    def isCorner(self, i):
        if i == 0:
            return 0b1100
        elif i == self.size - 1:
            return 0b0110
        elif i == (self.size ** 2 - self.size):
            return 0b1001
        elif i == (self.size ** 2 - 1):
            return 0b0011
        return 0
    
    def isEdge(self, i):
        if i < self.size:
            return 0b0100
        elif i % self.size == 0:
            return 0b1000
        elif (i + 1) % self.size == 0:
            return 0b0010
        elif (self.size ** 2 - i) <= self.size:
            return 0b0001
        return 0
        
    @staticmethod
    def board_frontier_range(size):
        for i in range(size):
            yield i
        
        curr = size
        for i in range(size - 2):
            yield curr
            yield curr + size - 1
            curr += size
            
        for i in range(size):
            yield curr + i
    
    def direction_index_offset(self, facing):
        i = []
        if facing & 0x8:
            i.append((-1, 0x8))
        if facing & 0x4:
            i.append((-self.size, 0x4))
        if facing & 0x2:
            i.append((1, 0x2))
        if facing & 0x1:
            i.append((self.size, 0x1))
        
        return i
        
    def lshift(self, x, size, amount):
        return ((x << amount) % (1 << size)) | (x >> (size - amount))
           
            
    def find_locks(self, i):
        piece = self.storage[i]
        
        piece_high = piece & 0xF0
        piece_low = piece & 0xF
        adjs_indeces = self.direction_index_offset(piece_low)
        
        if None in self.get_piece_adjs(i, piece):
            return
        
        if self.isCorner(i):
            if piece_high == 0b0010_0000: # V pieces
                self.lockPiece(i)
                return
            elif not piece_high: # F pieces
                k = d = 0
                adj_i = i + adjs_indeces[0][0]
                
                if self.isLocked(adj_i) and (self.storage[adj_i] & piece_low): # connecting pieces
                    self.lockPiece(i)
                    return
                
                if adj_i > 0 and adj_i < self.size - 1: # Non connecting
                    k = self.size
                    d = 0b1010_0001
                elif adj_i > self.size ** 2 - self.size and adj_i < self.size ** 2:
                    k = -self.size
                    d = 0b1010_0100
                elif adj_i % self.size == 0:
                    k = 1
                    d = 0b1010_0010
                elif (adj_i + 1) % self.size == 0:
                    k = -1
                    d = 0b1010_1000
                        
                if self.storage[i + k] & 0x30 == 0 or (self.storage[i + k] == (d | piece_low)):
                    self.lockPiece(i)
                    return
            return 
        w = self.isEdge(i)   
        if w:
            if piece_high == 0b0011_0000: # L pieces
                self.lockPiece(i)
            elif piece_high == 0b0001_0000: # B pieces
                self.lockPiece(i)
            elif piece_high == 0b0010_0000: # V pieces
                p_v = piece_low & 0b0101
                p_h = piece_low & 0b1010
                rev_p = self.lshift(piece_low, 4, 2)
                k = 0
                
                if i % self.size == 0 or (i + 1) % self.size == 0:
                    k = self.direction_index_offset(p_v)[0][0]
                else:
                    k = self.direction_index_offset(p_h)[0][0]
                    
                if self.isLocked(i + k) and (self.storage[i + k] ^ rev_p) & rev_p == rev_p & 0b0101 or \
                    self.isLocked(i - k) and not (self.storage[i - k] & (rev_p & 0b0101 | p_h)):
                        self.lockPiece(i)
            elif not piece_high: # F pieces
                k = adjs_indeces[0][0]
                m = self.lshift(piece_low, 4, 2) | w
                
                if self.isLocked(i + k) and (self.storage[i + k] ^ m) & m == w:
                    self.lockPiece(i)
                else:
                    for j in a.get((piece_low | w) ^ 0xF):
                        k_d = self.direction_index_offset(j)[0][0]
                        if self.lshift(j, 4, 2) == w: #Center dir
                            if self.storage[i + k_d] & 0xF != 0 and not (self.storage[i + k_d] & 0x88 == 0x80):
                                return
                        else: # Edge dir
                            if self.storage[i + k_d] & 0xF != 0 and not (self.storage[i + k_d] & 0xF == ((piece_low | w) ^ 0xF)):
                                return
                    # Lock if it completed for loop
                    self.lockPiece(i)
        else:
            y = False
            for k in adjs_indeces:
                if not self.isLocked(i + k[0]) or not (self.storage[i + k[0]] & k[1]):
                    y = False
                    break
                y = True
            if y:
                self.lockPiece(i)
    
    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """

        data = stdin.read().split()
        storage = [piece_to_byte[item] for item in data]
        # Numpy implementation (Slower but more memory efficient)
        # data = stdin.buffer.read().split()
        
        # storage = np.vectorize(Board.convert_piece)(np.frombuffer(b''.join(data), dtype=np.uint16))
        return Board(int(math.sqrt(len(storage))), storage)  


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        initial = PipeManiaState(board)
        super().__init__(initial)
        self.visited = []
        
    def generate_lockable_action(self, i, piece, board):
        piece_high = piece & 0xF0
        piece_low = piece & 0xF
        
        w_c = board.isCorner(i)
        if w_c: # Corner Pieces
            if piece_high == 0b0010_0000: # V piece
                return (i, 0x80 | piece_high | self.lshift(w_c, 4, 2))
            elif piece_high == 0: # F piece
                v_wall = w_c & 0b0101
                h_wall = w_c & 0b1010
                
                k_v = board.direction_index_offset(self.lshift(v_wall, 4, 2))[0]
                k_h = board.direction_index_offset(self.lshift(h_wall, 4, 2))[0]
                
                if board.storage[i + k_h[0]] & 0x30 == 0 or \
                    board.storage[i + k_h[0]] == (0xA0 | k_h[1] | k_v[1]) or \
                    board.storage[i + k_v[0]] & (0x80 | v_wall) == (0x80 | v_wall):
                        return (i, 0x80 | piece_high | k_v[1])
                
                elif board.storage[i + k_v[0]] & 0x30 == 0 or \
                    board.storage[i + k_v[0]] == (0xA0 | k_h[1] | k_v[1]) or \
                    board.storage[i + k_h[0]] & (0x80 | h_wall) == (0x80 | h_wall):
                        return (i, 0x80 | piece_high | k_h[1])
            return None
        w_e = board.isEdge(i)
        if w_e: # Edge Pieces
            if piece_high == 0b0011_0000: # L piece
                return (i, 0x80 | piece_high | (self.lshift(w_e, 4, 1) | self.rshift(w_e, 4, 1)))
            elif piece_high == 0b0001_0000: # B piece
                return (i, 0x80 | piece_high | (w_e ^ 0xF))
            elif piece_high == 0b0010_0000: # V piece
                k = board.direction_index_offset(self.lshift(w_e, 4, 1))[0]
                rev_k1 = self.lshift(k[1], 4, 2)
                
                if board.storage[i + k[0]] & (0x80 | rev_k1) == (0x80 | rev_k1) or \
                    board.storage[i - k[0]] & (0x80 | k[1]) == 0x80:
                    return (i, 0x80 | piece_high | self.lshift(w_e, 4, 2) | k[1])
                    
                elif board.storage[i + k[0]] & (0x80 | rev_k1) == 0x80 or \
                    board.storage[i - k[0]] & (0x80 | k[1]) == (0x80 | k[1]):
                    return (i, 0x80 | piece_high | self.lshift(w_e, 4, 2) | rev_k1)
            elif piece_high == 0: # F piece
                final_dir = w_e
                for k in board.direction_index_offset(w_e ^ 0xF):
                    rev_k1 = self.lshift(k[1], 4, 2)
                    if board.storage[i + k[0]] & 0x30 == 0 or board.storage[i + k[0]] & (0x80 | rev_k1) == 0x80:
                        final_dir |= k[1]
                final_dir ^= 0xF # Invert the bits
                if final_dir != 0 and (final_dir & (final_dir - 1)) == 0: # Check if only 1 bit is on
                    return (i, 0x80 | piece_high | final_dir)
        else: # Center Pieces
            if piece_high == 0b0011_0000: # L Piece
                k1, k2 = board.direction_index_offset(self.lshift(piece_low, 4, 1))
                if board.storage[i + k1[0]] & (0x80 | k2[1]) == (0x80 | k2[1]) and \
                    board.storage[i + k2[0]] & (0x80 | k1[1]) == (0x80 | k1[1]):
                        return (i, 0x80 | piece_high | k1[1] | k2[1])
            elif piece_high == 0b0001_0000: # B Piece
                final_dir = 0
                for k in board.direction_index_offset(0xF): # Get all directions
                    rev_k1 = self.lshift(k[1], 4, 2)
                    if board.storage[i + k[0]] & (0x80 | rev_k1) == 0x80:
                        return (i, 0x80 | piece_high | (k[1] ^ 0xF))
                    elif board.storage[i + k[0]] & (0x80 | rev_k1) == (0x80 | rev_k1):
                        final_dir |= k[1]
                        
                rev_final_dir = final_dir ^ 0xF
                if rev_final_dir != 0 and (rev_final_dir & (rev_final_dir - 1)) == 0:
                    return (i, 0x80 | piece_high | final_dir)
                
        return None         
        
    def lockable_actions(self, state: PipeManiaState):
        board = state.board
        lock_actions = []
        
        for i, piece in enumerate(board.storage):
            if board.isLocked(i):
                continue
            
            board.find_locks(i)
            
            if board.isLocked(i):
                continue
            
            action = self.generate_lockable_action(i, piece, board)
            if action:
                lock_actions.append(action)
            
        return lock_actions
    
    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        board: Board = state.board
        actions = []
        
        if self.isVisited(board):
            return actions
        
        self.visited.append(board.copy())
        
        actions = self.lockable_actions(state)
        
        if not actions: # Generate unfiltered actions if no lockable available
            for i, piece in enumerate(board.storage):
                if board.isLocked(i):
                    continue
                
                piece_high = (piece & 0xF0)
                piece_low = (piece & 0x0F)
                
                if piece_high != 0b0011_0000:
                    a = self.lshift(piece_low, 4, 1)
                    b = self.lshift(a, 4, 1)
                    c = self.lshift(b, 4, 1)
                    actions.extend([(i, piece_high | a), (i, piece_high | b), (i, piece_high | c)])
                else:
                    a = self.lshift(piece_low, 4, 1)
                    actions.append((i, piece_high | a))

        for action in actions:
            copy = state.board.copy()
            copy.change_piece(*action)
            
            for b in self.visited:
                if copy == b:
                    actions.remove(action)
                    break

        return actions
        

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        board = state.board.copy()
        board.change_piece(*action)
        
        return PipeManiaState(board)

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        
        # Nao deteta se houverem varias subcomponentes
        board = state.board
        for i, piece in enumerate(board.storage):
            # left, right, up, down
            adjs = board.adjacent_horizontal_values(0, 0, i) + board.adjacent_vertical_values(0, 0, i)
            
            p_left = piece & 0b0000_1000
            p_up = piece & 0b0000_0100
            p_right = piece & 0b0000_0010
            p_down = piece & 0b0000_0001
            
            if p_left and (not adjs[0] or not adjs[0] & 0b0000_0010):
                return False
            if p_up and (not adjs[2] or not adjs[2] & 0b0000_0001):
                return False
            if p_right and (not adjs[1] or not adjs[1] & 0b0000_1000):
                return False
            if p_down and (not adjs[3] or not adjs[3] & 0b0000_0100):
                return False
            
        return True
    
    def isVisited(self, board):
        for b in self.visited:
            if board == b:
                return True
        return False

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    def lshift(self, x, size, amount):
        return ((x << amount) % (1 << size)) | (x >> (size - amount))
    
    def rshift(self, x, size, amount):
        return ((x >> amount) | (x << (size - amount))) & ((1 << size) - 1)



if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    board = Board.parse_instance()
    
    problem = PipeMania(board)
    
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board.print())