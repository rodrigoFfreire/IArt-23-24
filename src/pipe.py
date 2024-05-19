# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

import math
import numpy as np
import random
from collections import deque

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

DIR_LEFT = 0b1000
DIR_UP = 0b0100
DIR_RIGHT = 0b0010
DIR_DOWN = 0b0001

PIECE_F = 0b0000_0000
PIECE_B = 0b0001_0000
PIECE_V = 0b0010_0000
PIECE_L = 0b0011_0000

LOCK_MASK = 0b1000_0000
PIECE_MASK = 0b0011_0000


class PipeManiaState:
    state_id = 0

    def __init__(self, board, last_index):
        self.board: Board = board
        self.id = PipeManiaState.state_id
        self.last_index = last_index
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, size: int, storage: list, exhausted) -> None:
        self.size = size
        self.storage = storage
        self.exhausted = exhausted

    def get_value(self, index) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if index < 0 or index >= self.size ** 2:
            return None
        return self.storage[index]

    def adjacent_vertical_values(self, index: int):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (
                self.storage[index - self.size] if index >= self.size else None,
                self.storage[index + self.size] if (self.size ** 2) - index > self.size else None
            )

    def adjacent_horizontal_values(self, index: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (
                self.storage[index - 1] if index % self.size != 0 else None,
                self.storage[index + 1] if (index + 1) % self.size != 0 else None
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
        return Board(self.size, self.storage.copy(), self.exhausted)

    def change_piece(self,index, new_piece):
        self.storage[index] = new_piece
        
    def __eq__(self, other) -> bool:
        return self.storage == other.storage
    
    def lockPiece(self, index, search_bad):
        if not search_bad:
            self.storage[index] |= 0x80
        return True
    
    def unlockPiece(self, index):
        self.storage[index] ^ 0x80
        
    def isLocked(self, index):
        return True if self.storage[index] & 0x80 else False
        
    def getAdjacentPieces(self, index, piece):
        h_adjs = self.adjacent_horizontal_values(index)
        v_adjs = self.adjacent_vertical_values(index)
        
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
                        
    def isLockable(self, i, search_bad):
        piece = self.storage[i]
        
        piece_high = piece & 0xF0
        piece_low = piece & 0xF
        adjs_indeces = self.direction_index_offset(piece_low)
        
        # If a piece has ends facing the border then its definitely not lockable
        if None in self.getAdjacentPieces(i, piece):
            return
        
        w_corner = self.isCorner(i)
        if w_corner:
            if piece_high == PIECE_V:
                return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_F:
                k = self.direction_index_offset(piece_low)[0]
                rev_k1 = self.lshift(k[1], 4, 2)
                if self.storage[i + k[0]] & (LOCK_MASK | rev_k1) == LOCK_MASK | rev_k1:
                    return self.lockPiece(i, search_bad)
                
                non_k = self.direction_index_offset((piece_low ^ 0xF) ^ w_corner)[0]
                if self.storage[i + non_k[0]] & (PIECE_MASK) == PIECE_F or \
                    self.storage[i + non_k[0]] == (w_corner ^ 0xF):
                        return self.lockPiece(i, search_bad)
                    
        w_edge = self.isEdge(i)
        if w_edge:
            if piece_high == PIECE_L:
                return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_B:
                return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_V:
                k = self.direction_index_offset((w_edge ^ 0xF) ^ piece_low)[0]
                rev_k1 = self.lshift(k[1], 4, 2)
                
                if self.storage[i - k[0]] & (LOCK_MASK | k[1]) == (LOCK_MASK | k[1]) or \
                    self.storage[i + k[0]] & (LOCK_MASK | rev_k1) == LOCK_MASK:
                        return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_F:
                k = adjs_indeces[0]
                rev_k1 = self.lshift(piece_low, 4, 2)
                non_k = self.direction_index_offset((piece_low | w_edge) ^ 0xF)
                
                if self.storage[i + k[0]] & (LOCK_MASK | rev_k1) == (LOCK_MASK | rev_k1):
                    return self.lockPiece(i, search_bad)
                elif (self.storage[i + non_k[0][0]] & PIECE_MASK == PIECE_F or \
                    self.storage[i + non_k[0][0]] & (LOCK_MASK | self.lshift(non_k[0][1], 4, 2)) == 0x80) and \
                    (self.storage[i + non_k[1][0]] & PIECE_MASK == PIECE_F or \
                    self.storage[i + non_k[1][0]] & (LOCK_MASK | self.lshift(non_k[1][1], 4, 2)) == 0x80):
                        return self.lockPiece(i, search_bad)
        else:
            if piece_high == PIECE_L:
                k = self.direction_index_offset(piece_low)
                rev_k = self.direction_index_offset(self.lshift(piece_low, 4, 1))
                
                if self.storage[i + k[0][0]] & (LOCK_MASK | k[1][1]) == (LOCK_MASK | k[1][1]) or \
                    self.storage[i + k[1][0]] & (LOCK_MASK | k[0][1]) == (LOCK_MASK | k[0][1]):
                        return self.lockPiece(i, search_bad)
                
                elif self.storage[i + rev_k[0][0]] & (LOCK_MASK | rev_k[1][1]) == LOCK_MASK or \
                    self.storage[i + rev_k[1][0]] & (LOCK_MASK | rev_k[0][1]) == LOCK_MASK:
                        return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_B:
                k = self.direction_index_offset(piece_low)
                non_k = self.direction_index_offset(piece_low ^ 0xF)
                rev_non_k1 = self.lshift(non_k[0][1], 4, 2)
                
                rev_k01 = self.lshift(k[0][1], 4, 2)
                rev_k11 = self.lshift(k[1][1], 4, 2)
                rev_k21 = self.lshift(k[2][1], 4, 2)
                
                if self.storage[i + non_k[0][0]] & (LOCK_MASK | rev_non_k1) == LOCK_MASK:
                    return self.lockPiece(i, search_bad)
                elif self.storage[i + k[0][0]] & (LOCK_MASK | rev_k01) == (LOCK_MASK | rev_k01) and \
                    self.storage[i + k[1][0]] & (LOCK_MASK | rev_k11) == (LOCK_MASK | rev_k11) and \
                    self.storage[i + k[2][0]] & (LOCK_MASK | rev_k21) == (LOCK_MASK | rev_k21):
                        return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_V:
                k = self.direction_index_offset(piece_low)
                non_k = self.direction_index_offset(piece_low ^ 0xF)
                
                rev_k01 = self.lshift(k[0][1], 4, 2)
                rev_k11 = self.lshift(k[1][1], 4, 2)
                
                if self.storage[i + k[0][0]] & (LOCK_MASK | rev_k01) == (LOCK_MASK | rev_k01) and \
                    (self.storage[i + k[1][0]] & (LOCK_MASK | rev_k11) == (LOCK_MASK | rev_k11) or self.storage[i - k[1][0]] & (0x80 | k[1][1]) == 0x80):
                        return self.lockPiece(i, search_bad)
                   
                elif (self.storage[i + k[1][0]] & (LOCK_MASK | rev_k11) == (LOCK_MASK | rev_k11)) and \
                    (self.storage[i + k[0][0]] & (LOCK_MASK | rev_k01) == (LOCK_MASK | rev_k01) or self.storage[i - k[0][0]] & (0x80 | k[0][1]) == 0x80):
                       return self.lockPiece(i, search_bad)
                elif self.storage[i - k[0][0]] & (LOCK_MASK | k[0][1]) == LOCK_MASK and \
                    self.storage[i - k[1][0]] & (LOCK_MASK | k[1][1]) == LOCK_MASK:
                        return self.lockPiece(i, search_bad)
            elif piece_high == PIECE_F:
                rev_k1_c = self.lshift(piece_low, 4, 2)
                
                non_k = self.direction_index_offset(piece_low ^ 0xF)
                
                if self.storage[i + adjs_indeces[0][0]] & (LOCK_MASK | rev_k1_c) == (LOCK_MASK | rev_k1_c):
                    return self.lockPiece(i, search_bad)
                elif (self.storage[i + non_k[0][0]] & PIECE_MASK == PIECE_F or self.storage[i + non_k[0][0]] & (LOCK_MASK | self.lshift(non_k[0][1], 4, 2)) == 0x80) and \
                    (self.storage[i + non_k[1][0]] & PIECE_MASK == PIECE_F or self.storage[i + non_k[1][0]] & (LOCK_MASK | self.lshift(non_k[1][1], 4, 2)) == 0x80) and \
                    (self.storage[i + non_k[2][0]] & PIECE_MASK == PIECE_F or self.storage[i + non_k[2][0]] & (LOCK_MASK | self.lshift(non_k[2][1], 4, 2)) == 0x80):
                        return self.lockPiece(i, search_bad)
            
                        
    def getAdjacentIndeces(self, i):
        h_adjs = self.adjacent_horizontal_values(i)
        v_adjs = self.adjacent_vertical_values(i)
        
        adjs = []
        if h_adjs[0] and not (h_adjs[0] & 0x80): adjs.append(i - 1)
        if v_adjs[0] and not (v_adjs[0] & 0x80): adjs.append(i - self.size)
        if h_adjs[1] and not (h_adjs[1] & 0x80): adjs.append(i + 1)
        if v_adjs[1] and not (v_adjs[1] & 0x80): adjs.append(i + self.size)
        
        return adjs
    
    def find_bad_locks(self, i):
        piece = self.storage[i]
        piece_low = piece & 0xF
        for k in self.direction_index_offset(0xF):
            rev_k1 = self.lshift(k[1], 4, 2)
            pk = self.get_value(i + k[0])
            if piece_low & k[1] == 0 and pk is None: continue
            
            if (piece_low & k[1] and (pk is None or pk & (0x80 | rev_k1) == 0x80)) or \
                (piece_low & k[1] == 0 and pk & (0x80 | rev_k1) == 0x80 | rev_k1):
                    return True
        return False
        
                        
    def find_locks(self, indeces: list, search_bad):
        min_i = self.size ** 2
        queue = indeces
        while queue:
            p = queue.pop()
            if self.isLocked(p):
                continue
            if self.isLockable(p, search_bad):
                if search_bad and self.find_bad_locks(p):
                    return -1
                min_i = min(min_i, p)
                queue.extend(self.getAdjacentIndeces(p))
                
        return min_i

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """

        data = stdin.read().split()
        storage = [piece_to_byte[item] for item in data]

        size = int(math.sqrt(len(storage)))
        indeces = [i for i in range(size ** 2)] # Generates all indeces from 0 -> (size - 1)
        
        board = Board(size, storage, False)
        board.find_locks(indeces, False) # Looks for all pieces that can be LOCK_MASK initially
        print("Locks:")
        out = []
        for i in range(len(board.storage)):
            out.append(str(board.isLocked(i))) # Remove lock bit
            if ((i + 1) % board.size != 0):
                out.append('\t')
            else:
                out.append('\n')
        print("".join(out))
        
        return board


class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        initial = PipeManiaState(board, 0)
        super().__init__(initial)
        self.visited = []
        
    def generate_lockable_action(self, i, board: Board):
        piece = board.storage[i]
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
                    elif board.storage[i + k[0]] & (0x80 | rev_k1) == (0x80 | rev_k1):
                        return (i, 0x80 | piece_high | k[1])
                final_dir ^= 0xF # Invert the bits
                if final_dir != 0 and (final_dir & (final_dir - 1)) == 0: # Check if only 1 bit is on
                    return (i, 0x80 | piece_high | final_dir)
        else: # Center Pieces
            if piece_high == 0b0011_0000: # L Piece
                k1, k2 = board.direction_index_offset(piece_low)
                rev_k1, rev_k2 = board.direction_index_offset(self.lshift(piece_low, 4, 1))
                
                if board.storage[i + rev_k1[0]] & (0x80 | rev_k2[1]) == (0x80 | rev_k2[1]) or \
                    board.storage[i + rev_k2[0]] & (0x80 | rev_k1[1]) == (0x80 | rev_k1[1]) or \
                    board.storage[i + k1[0]] & (0x80 | k2[1]) == 0x80 or \
                    board.storage[i + k2[0]] & (0x80 | k1[1]) == 0x80:
                        return (i, 0x80 | piece_high | rev_k1[1] | rev_k2[1])
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
            elif piece_high == 0b0010_0000: # V Piece
                y = n = 0
                for k in board.direction_index_offset(0xF): # Get all directions
                    rev_k1 = self.lshift(k[1], 4, 2)
                    if board.storage[i + k[0]] & (0x80 | rev_k1) == (0x80 | rev_k1):
                        y |= k[1]
                    elif board.storage[i + k[0]] & (0x80 | rev_k1) == 0x80:
                        n |= k[1]
                r = y | self.lshift(n, 4, 2)
                if r != 0 and r % 3 == 0: # Checks if it has 2 adjacent directions (valid v piece)
                    return (i, 0x80 | piece_high | r)
            elif piece_high == 0: # F piece
                final_dir = 0
                for k in board.direction_index_offset(0xF): # Get all directions
                    rev_k1 = self.lshift(k[1], 4, 2)
                    if board.storage[i + k[0]] & (0x80 | rev_k1) == (0x80 | rev_k1):
                        return (i, 0x80 | piece_high | k[1])
                    elif board.storage[i + k[0]] & (0x80 | rev_k1) == 0x80 or \
                        board.storage[i + k[0]] & 0x30 == 0:
                        final_dir |= k[1]
                
                rev_final_dir = final_dir ^ 0xF
                if rev_final_dir != 0 and (rev_final_dir & (rev_final_dir - 1)) == 0:
                    return (i, 0x80 | piece_high | rev_final_dir)
                
        return None
    
    def correctLastIndex(self, index, size):
        if index <= 0:
            return 0
        elif index < size:
            return index - 1
        else:
            return index - size
        
    
    def lockableAction(self, state: PipeManiaState):
        board = state.board
        
        last_index = self.correctLastIndex(state.last_index, board.size)    
            
        print("LAST INDEX:", last_index)
        for i in range(last_index, board.size ** 2):
            if board.isLocked(i):
                continue
            action = self.generate_lockable_action(i, board)
            if action is not None:
                if board.exhausted:
                    board.change_piece(*action)
                    if board.find_locks(board.getAdjacentIndeces(i), True) == -1:
                        return -1
                return action
           
        return None
    
    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        board: Board = state.board
        actions = []
        
        print("Expanding state:", state.id)
        print(board.print())
        
        if state.id == 1155:
            print("aaa")
        
        lock_action = self.lockableAction(state)
        if lock_action == -1:
            print("BACKTRACKING")
            return ()
        
        print("Locks:")
        out = []
        for i in range(len(board.storage)):
            out.append(str(board.isLocked(i))) # Remove lock bit
            if ((i + 1) % board.size != 0):
                out.append('\t')
            else:
                out.append('\n')
        print("".join(out))
        
        if lock_action is None:
            print("GENERATING RANDOM ACTIONS")
            board.exhausted = True
            for i, piece in enumerate(board.storage):
                if board.isLocked(i):
                    continue
                
                piece_high = (piece & 0xF0)
                piece_low = (piece & 0x0F)
                
                ly = ln = 0
                for k in board.direction_index_offset(0xF):
                    rev_k1 = self.lshift(k[1], 4, 2)
                    p = board.get_value(i + k[0])
                    if p is None or p & (0x80 | rev_k1) == 0x80:
                        ln |= k[1]
                    elif p & (0x80 | rev_k1) == (0x80 | rev_k1):
                        ly |= k[1]
                
                if piece_high != 0b0011_0000:
                    a = self.lshift(piece_low, 4, 1)
                    b = self.lshift(a, 4, 1)
                    c = self.lshift(b, 4, 1)
                    
                    p_a = 0x80 | piece_high | a
                    p_b = 0x80 | piece_high | b
                    p_c = 0x80 | piece_high | c
                    
                    if (piece_low & ly) == ly and (piece_low & ln) == 0:
                        actions.append((i, 0x80 | piece))
                    if (p_a & ly) == ly and (p_a & ln) == 0:
                        actions.append((i, p_a))
                    if (p_b & ly) == ly and (p_b & ln) == 0:
                        actions.append((i, p_b))
                    if (p_c & ly) == ly and (p_c & ln) == 0:
                        actions.append((i, p_c))
                    return actions
                else:
                    a = self.lshift(piece_low, 4, 1)
                    
                    p_a = 0x80 | piece_high | a
                    
                    if (piece_low & ly) == ly and (piece_low & ln) == 0:
                        actions.append((i, 0x80 | piece))
                    if (p_a & ly) == ly and (p_a & ln) == 0:
                        actions.append((i, p_a))
                    return actions
        else:
            actions = (lock_action,)
            
        print(actions)
        return actions


    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        board = state.board.copy()
        board.change_piece(*action)
        min_find = board.find_locks(board.getAdjacentIndeces(action[0]), False)
        
        return PipeManiaState(board, min(action[0], min_find))

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        
        # Nao deteta se houverem varias subcomponentes
        board = state.board
        for i, piece in enumerate(board.storage):
            # left, right, up, down
            adjs = board.adjacent_horizontal_values(i) + board.adjacent_vertical_values(i)
            
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
    print("Done")
    
    problem = PipeMania(board)
    goal_node = depth_first_tree_search(problem)
    
    print("Is goal?", problem.goal_test(goal_node.state))
    print(goal_node.state.board.print())