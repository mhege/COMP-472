# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

# Work to be done:
# Input valid game parameters.
# Change tic tac toe to represent values.
# Make the e1 heuristic to verify work.

import time

class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3
    colLabels = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}

    def __init__(self, recommend = True):

        self.input()
        self.initialize_game()
        self.recommend = recommend

    def input(self):

        # Initialize values for input verification
        board_check = True
        bloc_check = True
        self.bloc_posi = []
        bloc_posi_count = 0
        line_check = True
        depth_check = True
        time_check = True
        algo_check = True
        playmode_check = True

        # Get input for the size of the board
        while board_check:
            self.board_size = int(input("Enter the board size n: "))
            if self.board_size not in range(11):
                print("Enter a value between 0 and 10 inclusive")
            else:
                board_check = False

        # Get input for the number of blocs on the board
        while bloc_check:
            self.num_blocs = int(input("Enter the number of blocs on the board b: "))
            if self.num_blocs not in range(2*self.board_size + 1):
                print("Enter a value between 0 and " + str(2*self.board_size) + " inclusive")
            else:
                bloc_check = False

        # Get unique bloc positions for the number of blocs
        while bloc_posi_count != self.num_blocs:
            x_posi = input("Enter the column the bloc will go: ")
            y_posi = int(input("Enter the row the bloc will go: "))
            valid_posi = True
            if x_posi in self.colLabels:
                x_posi = self.colLabels[x_posi]
                if x_posi not in range(self.board_size + 1) or y_posi not in range(self.board_size + 1):
                    print("Enter a valid location on the board")
                else:
                    if self.bloc_posi:
                        for i, val in enumerate(self.bloc_posi):
                            if self.bloc_posi[i][0] == x_posi and self.bloc_posi[i][1] == y_posi:
                                print("This position has already been entered")
                                valid_posi = False
                                break
                        if valid_posi:
                            self.bloc_posi.append([x_posi, y_posi])
                            bloc_posi_count += 1
                        valid_posi = True
                    else:
                        self.bloc_posi.append([x_posi, y_posi])
                        bloc_posi_count += 1
            else:
                print("Enter a correct column indicator")

        # Get winning line size
        while line_check:
            self.line_size = int(input("Enter the winning line size: "))
            if self.line_size not in range(3, self.board_size+1):
                print("Enter a value between 3 and " + str(self.board_size) + " inclusive")
            else:
                line_check = False

        # Get max depth of adversarial search for both players
        while depth_check:
            depthP1 = int(input("Enter the max depth of the adversarial search for player 1: "))
            depthP2 = int(input("Enter the max depth of the adversarial search for player 2: "))
            if depthP1 <= 0 or depthP2 <= 0:
                print("Enter values that are positive and nonzero for both depths")
            else:
                depth_check = False
                self.max_depth = [depthP1, depthP2]

        # Get maximum amount of time the adversarial search can run if the player is an AI
        while time_check:
            self.max_AI_time = float(input("Enter the maximum amount of time the AI can think: "))
            if self.max_AI_time <= 0:
                print("Enter a positive non zero value")
            else:
                time_check = False

        # Get Algorithm to run
        while algo_check:
            self.algo = int(input("Enter either 1 (True) for Alphabeta or 0 (False) for Minimax: "))
            if self.algo not in range(2):
                print("Enter either 0 or 1")
            else:
                algo_check = False

        # Get play mode. Let 0: H-H, 1:H-AI, 2:AI-H, 3:AI-AI
        while playmode_check:
            self.play_mode = int(input("Enter the play mode (0: H-H, 1:H-AI, 2:AI-H, 3:AI-AI): "))
            if self.play_mode not in range(4):
                print("Enter a value that represents a valid play mode")
            else:
                playmode_check = False

    def initialize_game(self):

        self.current_state = []

        for i in range(self.board_size):
            temp_row = [] # Generates new variable address
            for j in range(self.board_size):
                temp_row.append('.')
            self.current_state.append(temp_row)

        # Player X always plays first
        self.player_turn = 'X'

    # Draw board must be changed so that it is in reference to the board size entered
    def draw_board(self):
        print()
        for y in range(0, 3):
            for x in range(0, 3):
                print(F'{self.current_state[x][y]}', end="")
            print()
        print()

    # Change is valid so that px and py are in reference to the board size entered
    # Have an else if for the possibility of there being a bloc on the space
    def is_valid(self, px, py):
        if px < 0 or px > 2 or py < 0 or py > 2:
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    # Is end must be completely reworked in reference to board size input
    # Win condition must be made in reference to input line size
    def is_end(self):
        # Vertical win
        for i in range(0, 3):
            if (self.current_state[0][i] != '.' and
                    self.current_state[0][i] == self.current_state[1][i] and
                    self.current_state[1][i] == self.current_state[2][i]):
                return self.current_state[0][i]
        # Horizontal win
        for i in range(0, 3):
            if self.current_state[i] == ['X', 'X', 'X']:
                return 'X'
            elif self.current_state[i] == ['O', 'O', 'O']:
                return 'O'
        # Main diagonal win
        if (self.current_state[0][0] != '.' and
                self.current_state[0][0] == self.current_state[1][1] and
                self.current_state[0][0] == self.current_state[2][2]):
            return self.current_state[0][0]
        # Second diagonal win
        if (self.current_state[0][2] != '.' and
                self.current_state[0][2] == self.current_state[1][1] and
                self.current_state[0][2] == self.current_state[2][0]):
            return self.current_state[0][2]
        # Is whole board full?
        for i in range(0, 3):
            for j in range(0, 3):
                # There's an empty field, we continue the game
                if self.current_state[i][j] == '.':
                    return None
        # It's a tie!
        return '.'

    # This is fine for now
    def check_end(self):
        self.result = self.is_end()
        # Printing the appropriate message if the game has ended
        if self.result is not None:
            if self.result == 'X':
                print('The winner is X!')
            elif self.result == 'O':
                print('The winner is O!')
            elif self.result == '.':
                print("It's a tie!")
            self.initialize_game()
        return self.result

    # This is fine for now
    def input_move(self):
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = int(input('enter the x coordinate: '))
            py = int(input('enter the y coordinate: '))
            if self.is_valid(px, py):
                return px, py
            else:
                print('The move is not valid! Try again.')

    # THis is fine for now
    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'
        return self.player_turn

    # Must be changed in reference to board size
    # Must include the presence of blocs
    def minimax(self, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = 2
        if max:
            value = -2
        x = None
        y = None
        result = self.is_end()
        if result == 'X':
            return -1, x, y
        elif result == 'O':
            return 1, x, y
        elif result == '.':
            return 0, x, y
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.minimax(max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.minimax(max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'
        return value, x, y

    # Must be changed in reference to board size
    # Must include the presence of blocs
    def alphabeta(self, alpha=-2, beta=2, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = 2
        if max:
            value = -2
        x = None
        y = None
        result = self.is_end()
        if result == 'X':
            return -1, x, y
        elif result == 'O':
            return 1, x, y
        elif result == '.':
            return 0, x, y
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.alphabeta(alpha, beta, max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.alphabeta(alpha, beta, max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'
                    if max:
                        if value >= beta:
                            return value, x, y
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return value, x, y
                        if value < beta:
                            beta = value
        return value, x, y

    # That's fine for now?
    def play(self ,algo=None ,player_x=None ,player_o=None):
        if algo is None:
            algo = self.ALPHABETA
        if player_x is None:
            player_x = self.HUMAN
        if player_o is None:
            player_o = self.HUMAN
        while True:
            self.draw_board()
            if self.check_end():
                return
            start = time.time()
            if algo == self.MINIMAX:
                if self.player_turn == 'X':
                    (_, x, y) = self.minimax(max=False)
                else:
                    (_, x, y) = self.minimax(max=True)
            else: # algo == self.ALPHABETA
                if self.player_turn == 'X':
                    (m, x, y) = self.alphabeta(max=False)
                else:
                    (m, x, y) = self.alphabeta(max=True)
            end = time.time()
            if (self.player_turn == 'X' and player_x == self.HUMAN) or \
                    (self.player_turn == 'O' and player_o == self.HUMAN):
                if self.recommend:
                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: x = {x}, y = {y}')
                (x, y) = self.input_move()
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
            self.current_state[x][y] = self.player_turn
            self.switch_player()


def main():
    g = Game(recommend=True)
    g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.AI)
    g.play(algo=Game.MINIMAX, player_x=Game.AI, player_o=Game.HUMAN)


if __name__ == "__main__":
    main()