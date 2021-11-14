# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time

class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3
    HH = 0
    HA = 1
    AH = 2
    AA = 3
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
            if self.board_size not in range(3, 11):
                print("Enter a value between 3 and 10 inclusive")
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
            depthP1 = int(input("Enter the max depth of the adversarial search for player 1 (X): "))
            depthP2 = int(input("Enter the max depth of the adversarial search for player 2 (O): "))
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
            self.algorithm = int(input("Enter either 1 (True) for Alphabeta or 0 (False) for Minimax: "))
            if self.algorithm not in range(2):
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

        for val in self.bloc_posi:
            self.current_state[val[0]][val[1]] = '*'

        # Player X always plays first
        self.player_turn = 'X'

    def draw_board(self,f=None):
        f.write("\n")
        f.write(F'Move #{self.turn_number}\n\n')
        print()
        print(F'Move #{self.turn_number}')
        for y in range(self.board_size):
            for x in range(self.board_size):
                f.write(F'{self.current_state[x][y]}')
                print(F'{self.current_state[x][y]}', end="")
            f.write("\n")
            print()
        f.write("\n")
        print()
        #self.e1()

    def is_valid(self, px, py):
        if px not in range(self.board_size) or py not in range(self.board_size):
            return False
        elif self.current_state[px][py] != '.':
            return False
        else:
            return True

    def e1(self):
        openingsX = 0
        openingsO = 0

        # Vertical opening
        for i in range(self.board_size):
            checkX = 0
            checkO = 0
            boolX = True
            boolO = True
            for j in range(self.board_size):

                if (self.current_state[j][i] == '.' or self.current_state[j][i] == 'X') and boolX:
                    checkX += 1
                else:
                    checkX = 0

                if (self.current_state[j][i] == '.' or self.current_state[j][i] == 'O') and boolO:
                    checkO += 1
                else:
                    checkO = 0

                if checkX == self.line_size:
                    boolX = False
                    openingsX += 1

                if checkO == self.line_size:
                    boolO = False
                    openingsO += 1

        # Horizontal opening
        for j in range(self.board_size):
            checkX = 0
            checkO = 0
            boolX = True
            boolO = True
            for i in range(self.board_size):

                if (self.current_state[j][i] == '.' or self.current_state[j][i] == 'X') and boolX:
                    checkX += 1
                else:
                    checkX = 0

                if (self.current_state[j][i] == '.' or self.current_state[j][i] == 'O') and boolO:
                    checkO += 1
                else:
                    checkO = 0

                if checkX == self.line_size:
                    boolX = False
                    openingsX += 1

                if checkO == self.line_size:
                    boolO = False
                    openingsO += 1

        # Main diagonal
        # Takes into account off-diagonals (Top half left to right)
        for j in range((self.board_size + 1)-self.line_size):
            checkX = 0
            checkO = 0
            boolX = True
            boolO = True
            for i in range(self.board_size - j):

                if (self.current_state[i][i+j] == '.' or self.current_state[i][i+j] == 'X') and boolX:
                    checkX += 1
                else:
                    checkX = 0

                if (self.current_state[i][i+j] == '.' or self.current_state[i][i+j] == 'O') and boolO:
                    checkO += 1
                else:
                    checkO = 0

                if checkX == self.line_size:
                    boolX = False
                    openingsX += 1

                if checkO == self.line_size:
                    boolO = False
                    openingsO += 1

        # Second diagonal
        # Takes into account off-diagonals (Top half right to left)
        for j in range((self.board_size+1)-self.line_size):
            checkX = 0
            checkO = 0
            boolX = True
            boolO = True
            for i in range(self.board_size - j):

                if (self.current_state[i][self.board_size - 1 - i - j] == '.'
                    or self.current_state[i][self.board_size - 1 - i - j] == 'X') and boolX:
                    checkX += 1
                else:
                    checkX = 0

                if (self.current_state[i][self.board_size - 1 - i - j] == '.'
                    or self.current_state[i][self.board_size - 1 - i - j] == 'O') and boolO:
                    checkO += 1
                else:
                    checkO = 0

                if checkX == self.line_size:
                    boolX = False
                    openingsX += 1

                if checkO == self.line_size:
                    boolO = False
                    openingsO += 1

        # Need to account for off-diagonals for board size > line size
        if self.board_size > self.line_size:

            # Off-diagonal left side
            # Excludes main diagonal
            for j in range(self.board_size - self.line_size):
                checkX = 0
                checkO = 0
                boolX = True
                boolO = True
                for i in range(self.board_size - 1 - j):

                    if (self.current_state[i + j + 1][i] == '.' or self.current_state[i + j + 1][i] == 'X') and boolX:
                        checkX += 1
                    else:
                        checkX = 0

                    if (self.current_state[i + j + 1][i] == '.' or self.current_state[i + j + 1][i] == 'O') and boolO:
                        checkO += 1
                    else:
                        checkO = 0

                    if checkX == self.line_size:
                        boolX = False
                        openingsX += 1

                    if checkO == self.line_size:
                        boolO = False
                        openingsO += 1

            # Off-diagonal right side
            # Excludes second diagonal
            for j in range(self.board_size - self.line_size):
                checkX = 0
                checkO = 0
                boolX = True
                boolO = True
                for i in range(self.board_size - 1 - j):

                    if self.current_state[i + j + 1][self.board_size - 1 - i] == '.' \
                            or self.current_state[i + j + 1][self.board_size - 1 - i] == 'X' and boolX:
                        checkX += 1
                    else:
                        checkX = 0

                    if self.current_state[i + j + 1][self.board_size - 1 - i] == '.' \
                            or self.current_state[i + j + 1][self.board_size - 1 - i] == 'O' and boolO:
                        checkO += 1
                    else:
                        checkO = 0

                    if checkX == self.line_size:
                        boolX = False
                        openingsX += 1

                    if checkO == self.line_size:
                        boolO = False
                        openingsO += 1

        #print(openingsX)
        #print(openingsO)
        return openingsX - openingsO

    def e2(self):

        lengths = list(range(1, self.board_size+1))
        lengths.reverse()
        values = [10 ** length for length in lengths]

        totalX = 0
        totalO = 0

        # Vertical opening
        for i in range(self.board_size):
            valueX = 0
            valueO = 0

            for j in range(self.board_size):

                if self.current_state[j][i] == 'X':
                    valueX += 1
                elif self.current_state[j][i] == 'O':
                    valueO += 1

            for l in range(len(lengths)):
                if valueX == lengths[l]:
                    totalX += values[l]
                    break
                elif valueO == lengths[l]:
                    totalO += values[l]
                    break

        # Horizontal opening
        for j in range(self.board_size):
            valueX = 0
            valueO = 0

            for i in range(self.board_size):

                if self.current_state[j][i] == 'X':
                    valueX += 1
                elif self.current_state[j][i] == 'O':
                    valueO += 1

            for l in range(len(lengths)):
                if valueX == lengths[l]:
                    totalX += values[l]
                    break
                elif valueO == lengths[l]:
                    totalO += values[l]
                    break

        # Main diagonal
        # Takes into account off-diagonals (Top half left to right)
        for j in range((self.board_size + 1)-self.line_size):
            valueX = 0
            valueO = 0

            for i in range(self.board_size - j):

                if self.current_state[i][i+j] == 'X':
                    valueX += 1
                elif self.current_state[i][i+j]:
                    valueO += 1

            for l in range(len(lengths)):
                if valueX == lengths[l]:
                    totalX += values[l]
                    break
                elif valueO == lengths[l]:
                    totalO += values[l]
                    break

        # Second diagonal
        # Takes into account off-diagonals (Top half right to left)
        for j in range((self.board_size+1)-self.line_size):
            valueX = 0
            valueO = 0

            for i in range(self.board_size - j):

                if self.current_state[i][self.board_size - 1 - i - j] == 'X':
                    valueX += 1
                elif self.current_state[i][self.board_size - 1 - i - j] == 'O':
                    valueO += 1

            for l in range(len(lengths)):
                if valueX == lengths[l]:
                    totalX += values[l]
                    break
                elif valueO == lengths[l]:
                    totalO += values[l]
                    break

        # Need to account for off-diagonals for board size > line size
        if self.board_size > self.line_size:

            # Off-diagonal left side
            # Excludes main diagonal
            for j in range(self.board_size - self.line_size):
                valueX = 0
                valueO = 0

                for i in range(self.board_size - 1 - j):

                    if self.current_state[i + j + 1][i] == 'X':
                        valueX += 1
                    elif self.current_state[i + j + 1][i] == 'O':
                        valueO += 1

                for l in range(len(lengths)):
                    if valueX == lengths[l]:
                        totalX += values[l]
                        break
                    elif valueO == lengths[l]:
                        totalO += values[l]
                        break


            # Off-diagonal right side
            # Excludes second diagonal
            for j in range(self.board_size - self.line_size):
                valueX = 0
                valueO = 0

                for i in range(self.board_size - 1 - j):

                    if self.current_state[i + j + 1][self.board_size - 1 - i] == 'X':
                        valueX += 1
                    elif self.current_state[i + j + 1][self.board_size - 1 - i] == 'O':
                        valueO += 1

                for l in range(len(lengths)):
                    if valueX == lengths[l]:
                        totalX += values[l]
                        break
                    elif valueO == lengths[l]:
                        totalO += values[l]
                        break

        return totalX - totalO

    def is_end(self):

        # Vertical win
        for i in range(self.board_size):
            lineWin = 0
            for j in range(self.board_size - 1):
                if self.current_state[j][i] == '.' or self.current_state[j][i] == '*' \
                        or self.current_state[j][i] != self.current_state[j + 1][i]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.line_size-1:
                    return self.current_state[j][i]

        # Horizontal win
        for j in range(self.board_size):
            lineWin = 0
            for i in range(self.board_size - 1):
                if self.current_state[j][i] == '.' or self.current_state[j][i] == '*' \
                        or self.current_state[j][i] != self.current_state[j][i + 1]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.line_size-1:
                    return self.current_state[j][i]

        # Main diagonal win
        # Takes into account off-diagonals (Top half left to right)
        for j in range((self.board_size + 1)-self.line_size):
            lineWin = 0
            for i in range(self.board_size - 1 - j):
                if self.current_state[i][i+j] == '.' or self.current_state[i][i+j] == '*' \
                        or self.current_state[i][i+j] != self.current_state[i + 1][i + j + 1]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.line_size-1:
                    return self.current_state[i][i+j]

        # Second diagonal win
        # Takes into account off-diagonals (Top half right to left)
        for j in range((self.board_size+1)-self.line_size):
            lineWin = 0
            for i in range(self.board_size - 1 - j):
                if self.current_state[i][self.board_size - 1 - i - j] == '.' \
                    or self.current_state[i][self.board_size - 1 - i - j] == '*' \
                        or self.current_state[i][self.board_size - 1 - i - j] \
                        != self.current_state[i + 1][self.board_size - 1 - (i + 1) - j]:
                    lineWin = 0
                else:
                    lineWin += 1

                if lineWin == self.line_size-1:
                    return self.current_state[i][self.board_size - 1 - i - j]

        # Need to account for off-diagonals for board size > line size
        if self.board_size > self.line_size:

            # Off-diagonal left side
            # Excludes main diagonal
            for j in range(self.board_size-self.line_size):
                lineWin = 0
                for i in range(self.board_size - 2 - j):
                    if self.current_state[i + j + 1][i] == '.' or self.current_state[i + j + 1][i] == '*' \
                            or self.current_state[i + j + 1][i] != self.current_state[i + j + 2][i + 1]:
                        lineWin = 0
                    else:
                        lineWin += 1

                    if lineWin == self.line_size-1:
                        return self.current_state[i + j + 1][i]

            # Off-diagonal right side
            # Excludes second diagonal
            for j in range(self.board_size-self.line_size):
                lineWin = 0
                for i in range(self.board_size - 2 - j):
                    if self.current_state[i + j + 1][self.board_size - 1 - i] == '.' \
                            or self.current_state[i + j + 1][self.board_size - 1 - i] == '*' \
                            or self.current_state[i + j + 1][self.board_size - 1 - i] != self.current_state[i + j + 2][self.board_size - 1 - (i + 1)]:
                        lineWin = 0
                    else:
                        lineWin += 1

                    if lineWin == self.line_size-1:
                        return self.current_state[i + j + 1][self.board_size - 1 - i]

        # Is whole board full?
        for i in range(self.board_size):
            for j in range(self.board_size):
                # There's an empty field, we continue the game
                if self.current_state[i][j] == '.':
                    return None
        # It's a tie!
        return '.'

    def check_end(self, f = None):
        self.result = self.is_end()
        # Printing the appropriate message if the game has ended
        if self.result is not None:
            if self.result == 'X':
                f.write('The winner is X!')
                print('The winner is X!')
            elif self.result == 'O':
                f.write('The winner is O!')
                print('The winner is O!')
            elif self.result == '.':
                f.write("It's a tie!")
                print("It's a tie!")
            self.initialize_game()
        return self.result

    def input_move(self):
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = input('enter the x coordinate: ')
            py = int(input('enter the y coordinate: '))

            if px in self.colLabels:
                px = self.colLabels[px]
            else:
                print('The move is not valid! Try again.')
                continue

            if self.is_valid(px, py):
                return px, py
            else:
                print('The move is not valid! Try again.')


    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'
        return self.player_turn


    def minimax(self, max=False, depth = 0):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -inf - win for 'X'
        # 0  - a tie
        # inf  - loss for 'X'
        # We're initially setting it to inf++ or -inf-- as worse than the worst case:

        value = 10000000
        if max:
            value = -10000000
        x = None
        y = None
        result = self.is_end()
        if round(time.time() - self.currentTime,7) >= (9.5/10)*self.max_AI_time:
            return self.e1(), x, y
        elif result == 'X':
            return -1000000, x, y
        elif result == 'O':
            return 1000000, x, y
        elif result == '.':
            return 0, x, y
        elif self.max_depth[0] == depth + 1:
            return self.e1(), x, y
        elif self.max_depth[1] == depth + 1:
            return self.e1(), x, y
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.minimax(max=False, depth=depth+1)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.minimax(max=True,depth=depth+1)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = '.'

        return value, x, y


    def alphabeta(self, alpha=-2, beta=2, max=False, depth=0):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -inf - win for 'X'
        # 0  - a tie
        # inf  - loss for 'X'
        # We're initially setting it to inf++  or -inf-- as worse than the worst case:
        
        value = 10000000
        if max:
            value = -10000000
        x = None
        y = None
        result = self.is_end()
        if round(time.time() - self.currentTime,7) >= (9.5/10)*self.max_AI_time:
            self.number_of_evaluted_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1

            return self.e1(), x, y
        elif result == 'X':
            return -1000000, x, y
        elif result == 'O':
            return 1000000, x, y
        elif result == '.':
            return 0, x, y
        elif self.max_depth[0] == depth + 1:
            self.number_of_evaluted_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1

            return self.e1(), x, y
        elif self.max_depth[1] == depth + 1:
            self.number_of_evaluted_nodes += 1

            if str(depth) in self.depth_dict:
                self.depth_dict[str(depth)] += 1
            else:
                self.depth_dict[str(depth)] = 1

            return self.e1(), x, y
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.current_state[i][j] == '.':
                    if max:
                        self.current_state[i][j] = 'O'
                        (v, _, _) = self.alphabeta(alpha, beta, max=False, depth=depth+1)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = 'X'
                        (v, _, _) = self.alphabeta(alpha, beta, max=True, depth=depth+1)
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


    def play(self ,algo=None ,player_x=None ,player_o=None, f=None):
        if algo is None:
            algo = self.ALPHABETA
        if player_x is None:
            player_x = self.HUMAN
        if player_o is None:
            player_o = self.HUMAN

        self.turn_number = 1

        while True:
            self.currentTime = 0
            self.draw_board(f = f)
            if self.check_end(f = f):
                return
            start = time.time()
            self.currentTime = time.time()

            self.number_of_evaluted_nodes = 0
            self.depth_dict = {}

            if algo == self.MINIMAX:
                if self.player_turn == 'X':
                    (_, x, y) = self.minimax(max=False)
                else:
                    (_, x, y) = self.minimax(max=True)
            else:
                if self.player_turn == 'X':
                    (m, x, y) = self.alphabeta(max=False)
                else:
                    (m, x, y) = self.alphabeta(max=True)
            end = time.time()

            f.write(F'Evaluation time: {round(end - start, 7)}s\n')
            f.write(F'Number of evaluated nodes: {self.number_of_evaluted_nodes}\n')
            f.write(F'Evaluations by depth: {self.depth_dict}\n')

            if self.number_of_evaluted_nodes > 0:
                sum = 0
                for key in self.depth_dict:
                    sum += int(key) * self.depth_dict[key]
                f.write(F'Average evaluation depth: {sum / self.number_of_evaluted_nodes}\n')

            if (self.player_turn == 'X' and player_x == self.HUMAN) or \
                    (self.player_turn == 'O' and player_o == self.HUMAN):
                if self.recommend:
                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: '
                          F'x = {list(self.colLabels.keys())[list(self.colLabels.values()).index(x)]}, '
                          F'y = {y}')
                (x, y) = self.input_move()
                f.write(F'Player {self.player_turn} plays: ' 
                        F'x = {x} '
                        F'y = {y}\n')
            if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
                print(F'Evaluation time: {round(end - start, 7)}s')
                f.write(F'Player {self.player_turn} under AI control plays: '
                      F'x = {list(self.colLabels.keys())[list(self.colLabels.values()).index(x)]}, '
                      F'y = {y}\n')
                print(F'Player {self.player_turn} under AI control plays: '
                      F'x = {list(self.colLabels.keys())[list(self.colLabels.values()).index(x)]}, '
                      F'y = {y}')
            self.current_state[x][y] = self.player_turn
            self.switch_player()
            self.turn_number += 1


    def getPlaymode(self):
        return self.play_mode

    def getAlgorithm(self):
        return self.algorithm

def main():
    g = Game(recommend=True)

    file_name = "gameTrace-{}{}{}{}.txt".format(g.board_size,g.num_blocs,g.line_size,int(g.max_AI_time))

    f = open(file_name, "w")
    f.write("n={} b={} s={} t={}\n".format(g.board_size,g.num_blocs,g.line_size,g.max_AI_time))
    f.write("blocks={}\n\n".format(g.bloc_posi))

    if g.play_mode == 0:
        f.write("Player 1: Human d={} a={} e1\n".format(g.max_depth[0], bool(g.algorithm)))
        f.write("Player 2: Human d={} a={} e1\n".format(g.max_depth[1], bool(g.algorithm)))
    elif g.play_mode == 1:
        f.write("Player 1: Human d={} a={} e1\n".format(g.max_depth[0], bool(g.algorithm)))
        f.write("Player 2: AI d={} a={} e1\n".format(g.max_depth[1], bool(g.algorithm)))
    elif g.play_mode == 2:
        f.write("Player 1: AI d={} a={} e1\n".format(g.max_depth[0], bool(g.algorithm)))
        f.write("Player 2: Human d={} a={} e1\n".format(g.max_depth[1], bool(g.algorithm)))
    elif g.play_mode == 3:
        f.write("Player 1: AI d={} a={} e1\n".format(g.max_depth[0], bool(g.algorithm)))
        f.write("Player 2: AI d={} a={} e1\n".format(g.max_depth[1], bool(g.algorithm)))


    if Game.getAlgorithm(g) == 0:
        if Game.getPlaymode(g) == Game.HH:
            g.play(algo=Game.MINIMAX, player_x=Game.HUMAN, player_o=Game.HUMAN, f = f)
        elif Game.getPlaymode(g) == Game.HA:
            g.play(algo=Game.MINIMAX, player_x=Game.HUMAN, player_o=Game.AI, f = f)
        elif Game.getPlaymode(g) == Game.AH:
            g.play(algo=Game.MINIMAX, player_x=Game.AI, player_o=Game.HUMAN, f = f)
        else:
            g.play(algo=Game.MINIMAX, player_x=Game.AI, player_o=Game.AI, f = f)
    else:
        if Game.getPlaymode(g) == Game.HH:
            g.play(algo=Game.ALPHABETA, player_x=Game.HUMAN, player_o=Game.HUMAN, f = f)
        elif Game.getPlaymode(g) == Game.HA:
            g.play(algo=Game.ALPHABETA, player_x=Game.HUMAN, player_o=Game.AI, f = f)
        elif Game.getPlaymode(g) == Game.AH:
            g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.HUMAN, f = f)
        else:
            g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.AI, f = f)

    f.close()

if __name__ == "__main__":
    main()