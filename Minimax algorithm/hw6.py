from enum import Enum
from typing import List, Optional
import copy

class Move:
    #constant variable required argument
    INVALID_COORDINATE = -1
    
    def __init__(self, value: float, row: int = INVALID_COORDINATE, col: int = INVALID_COORDINATE):
        self.row = row
        self.col = col
        self.value = value


#enumerated type
class Player(Enum):
    X = "X"
    O = "O"
    
    def opposite(self):
        return Player.O if self == Player.X else Player.X



class GameState:
    def __init__(self):
        #2D array (list)  3 x 3 board full of None
        self.board = [[None for _ in range(3)] for _ in range(3)]
    
    def game_over(self) -> bool:
        # return the player who won the game 
        if self.winner() is not None:
            # print("None , no one won the game")
            return True
            
        #Check if board is full
        #if board is full which means the game is over
        #check if line by line
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    return False
        return True
    
    
    # return the player who won the game 
    # it return None
    def winner(self) -> Optional[Player]:
        #rows
        for row in self.board:
            if row[0] is not None and row[0] == row[1] == row[2]:
                return row[0]
        
        #columns
        for col in range(3):
            if (self.board[0][col] is not None and 
                self.board[0][col] == self.board[1][col] == self.board[2][col]):
                return self.board[0][col]
        
        #diagonals
        if (self.board[0][0] is not None and 
            self.board[0][0] == self.board[1][1] == self.board[2][2]):
            return self.board[0][0]
            
        if (self.board[0][2] is not None and 
            self.board[0][2] == self.board[1][1] == self.board[2][0]):
            return self.board[0][2]
        
        return None
    
    #representation of the board state
    def __str__(self) -> str:
        result = ""
        for row in self.board:
            for cell in row:
                if cell is None:
                    result += "- "
                else:
                    result += cell.value + " "
            result += "\n"
        return result
    
    #given position on the board
    def spot(self, row: int, col: int) -> Optional[Player]:
        return self.board[row][col]
    
    #return a new GameState
    #if the spot was already taken, return None 
    def move(self, row: int, col: int, player: Player) -> Optional['GameState']:
        
        if self.board[row][col] is not None:
            return None
            
        new_state = copy.deepcopy(self)
        new_state.board[row][col] = player
        return new_state
    
    def get_empty_spots(self):
        empty_spots = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    empty_spots.append((row, col))
        return empty_spots

#Minimax algorithm
class TicTacToeSolver:
    #returns the best move for this player to make given the current state.
    def find_best_move(self, state: GameState, player: Player) -> Move:
        alpha = float('-inf')
        beta = float('inf')
        return self.solve_my_move(state, alpha, beta, player)
    #return the score for the player whose score maximize
    def solve_my_move(self, state: GameState, alpha: float, beta: float, player: Player) -> Move:
        #base case
        if state.game_over():
            winner = state.winner()
            if winner is None:
                return Move(0)  #Draw
            elif winner == player:
                return Move(1)  #Win
            else:
                return Move(-1)  #Loss
        
        best_move = None
        
        for row, col in state.get_empty_spots():
            new_state = state.move(row, col, player)
            if new_state is None:
                continue
                
            child = self.solve_opponent_move(new_state, alpha, beta, player)
            
            #Update
            if best_move is None or child.value > best_move.value:
                best_move = Move(child.value, row, col)
            
            alpha = max(alpha, best_move.value)
            if beta <= alpha:
                break  #cut-off
        
        return best_move
    
    def solve_opponent_move(self, state: GameState, alpha: float, beta: float, player: Player) -> Move:
        #base case
        if state.game_over():
            winner = state.winner()
            if winner is None:
                return Move(0)  #Draw
            elif winner == player:
                return Move(1)  #Win
            else:
                return Move(-1)  #Loss
        
        best_move = None
        
        #try each empty spot
        for row, col in state.get_empty_spots():
            new_state = state.move(row, col, player.opposite())
            if new_state is None:
                continue
                
            child = self.solve_my_move(new_state, alpha, beta, player)
            
            #update best move 
            if best_move is None or child.value < best_move.value:
                best_move = Move(child.value, row, col)
            
            beta = min(beta, best_move.value)
            if beta <= alpha:
                break  #cut-off
        
        return best_move

def main():
    state = GameState()
    solver = TicTacToeSolver()
    current_player = Player.X
    
    print("Initial state:")
    print(state)
    
    while not state.game_over():
        # Find and make best move
        best_move = solver.find_best_move(state, current_player)
        state = state.move(best_move.row, best_move.col, current_player)
        
        print(f"\nPlayer {current_player.value} moves to ({best_move.row}, {best_move.col})")
        print(state)
        
        #switch players
        current_player = current_player.opposite()
    
    #print res
    winner = state.winner()
    if winner is None:
        print("Game ended in a draw!")
    else:
        print(f"Player {winner.value} wins!")

if __name__ == "__main__":
    main()