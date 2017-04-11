import random
import sys
import math
# import unittest
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # nopep8
from Player import Player
import Constants as c
from Construction import CONSTR_STATS, Construction
from Ant import UNIT_STATS, Ant
from Move import Move
from GameState import addCoords, subtractCoords, GameState
import AIPlayerUtils as utils
from Location import Location
from Inventory import Inventory
from Building import Building


class AIPlayer(Player):
    """
    Description:
        The responsibility of this class is to interact with the game
        by deciding a valid move based on a given game state. This class has
        methods that will be implemented by students in Dr. Nuxoll's AI course.

    Variables:
        playerId - The id of the player.
    """

    def __init__(self, inputPlayerId):
        """
        Creates a new Player

        Parameters:
            inputPlayerId - The id to give the new player (int)
        """
        super(AIPlayer, self).__init__(inputPlayerId, "Mr. Brain")
        self.initialWeights = [0.5] * 26
        self.outputWeights = [0.5] * 17
        self.perceptronBias = [0.5] * 17

    @staticmethod
    def neural_net(input_list):
        threshold = 1.5
        self.initialWeights[0]*input_list[0]


    """
    Description:
        Maps a set of inputs from a GameState into a list, that will then
        be input into the Neural Network.

    Parameters:
        state - GameState to score.
    """

    @staticmethod
    def map_input(state):

        input_list = [] * 12  # init list to be rtn'd

        enemy_id = abs(state.whoseTurn - 1)
        our_inv = utils.getCurrPlayerInventory(state)
        enemy_inv = [
            inv for inv in state.inventories if inv.player == enemy_id].pop()
        GOOD = 1
        BAD = 0
        we_win = 1.0
        enemy_win = 0.0
        our_food = our_inv.foodCount
        enemy_food = enemy_inv.foodCount
        food_difference = abs(our_food - enemy_food)
        our_anthill = our_inv.getAnthill()
        our_tunnel = our_inv.getTunnels()[0]
        enemy_anthill = enemy_inv.getAnthill()
        our_queen = our_inv.getQueen()
        enemy_queen = enemy_inv.getQueen()
        food_drop_offs = [our_tunnel.coords]
        food_drop_offs.append(our_anthill.coords)

        # input_list[0]: If our number of food is good or bad
        # Downside: incentivises AI to not spend food until it has
        #           1 over however many it takes to make an ant

        input_list.append(GOOD if food_difference > 0 else BAD)
        # input_list[1]: If our workers are carrying or depositing food it's good
        our_workers = [ant for ant in our_inv.ants if ant.type == c.WORKER]

        carrying_workers = [ant for ant in our_workers if ant.carrying]

        dropping_off = [
            ant for ant in our_workers if ant.coords in food_drop_offs and ant.carrying]

        input_list.append(GOOD if len(dropping_off) != 0 or len(carrying_workers) != 0 else BAD)

        # input_list[2]: If our workers are moving in a constructive manner
        movement_points = 0
        food_move = 100
        for ant in our_workers:
            ant_x = ant.coords[0]
            ant_y = ant.coords[1]
            for enemy in enemy_inv.ants:
                if ((abs(ant_x - enemy.coords[0]) > 3) and
                        (abs(ant_y - enemy.coords[1]) > 3)):
                    movement_points += 60
            if ant.carrying and ant not in dropping_off:
                # Good if carrying ants move toward a drop off.
                movement_points += food_move

                for dist in range(2, 4):
                    for dropoff in food_drop_offs:
                        if ((abs(ant_x - dropoff[0]) < dist) and
                                (abs(ant_y - dropoff[1]) < dist)):
                            movement_points += food_move - (dist * 25)
        input_list.append(GOOD if movement_points >= 160 else BAD)  # testing for now, can be changed upwards

        # input_list[3]: if we have an equal or greater amount of ants
        input_list.append(GOOD if len(our_inv.ants) >= len(enemy_inv.ants) else BAD)

        # input_list[4]: make sure we don't have too many workers
        enemy_workers = [ant for ant in enemy_inv.ants if ant.type == c.WORKER]
        input_list.append(GOOD if len(our_workers) < 3 else BAD)

        # input_list[5]: make sure workers don't leave friendly half of board
        our_range = [(x, y) for x in xrange(10) for y in xrange(5)]
        input_list.append(BAD if len([ant for ant in our_workers if ant.coords not in our_range]) != 0 else GOOD)

        # input_list[6]: make sure offensive ants are offensive
        # Let's just say each ant is worth 20x its cost for now
        offensive = [c.SOLDIER, c.R_SOLDIER, c.DRONE]
        our_offense = [ant for ant in our_inv.ants if ant.type in offensive]
        enemy_offense = [
            ant for ant in enemy_inv.ants if ant.type in offensive]

        offense_points = 0
        for ant in our_offense:
            ant_x = ant.coords[0]
            ant_y = ant.coords[1]
            attack_move = 160
            offense_points += UNIT_STATS[ant.type][c.COST] * 20  # every units cost * 20 ~ 60
            # good if on enemy anthill
            if ant.coords == enemy_anthill.coords:
                offense_points += 100  # ~ 160
            for enemy_ant in enemy_inv.ants:
                enemy_x = enemy_ant.coords[0]
                enemy_y = enemy_ant.coords[1]
                x_dist = abs(ant_x - enemy_x)
                y_dist = abs(ant_y - enemy_y)

                # good if attacker ant attacks
                if x_dist + y_dist == 1:
                    offense_points += attack_move * 2  # ~480

                # weighted more if closer to attacking
                for dist in xrange(1, 8):
                    if x_dist < dist and y_dist < dist:
                        offense_points += attack_move - (dist * 20)  # 160 - (3 * 20) = 100 ~580
        input_list.append(GOOD if offense_points >= 290 else BAD)

        # input_list[7]: Stop building if we have more than 5 ants
        input_list.append(BAD if our_inv.ants > 5 else GOOD)

        # input_list[8]: Queen healths, big deal, it's being attacked it's bad
        input_list.append(BAD if our_queen is not None and our_queen.health < 8 else GOOD)

        # input_list[9]: Stay off food_drop_offs and away from the front lines

        attack_points = 0

        if our_queen is not None:
            queen_coords = our_queen.coords
            input_list.append(BAD if queen_coords in food_drop_offs or queen_coords[1] > 2 else GOOD)

            # input_list[10]: queen attacks if under threat
            for enemy_ant in enemy_inv.ants:
                enemy_x = enemy_ant.coords[0]
                enemy_y = enemy_ant.coords[1]
                x_dist = abs(queen_coords[0] - enemy_x)
                y_dist = abs(queen_coords[1] - enemy_y)

                if (x_dist + y_dist) == 1:
                    attack_points += 200

        input_list.append(GOOD if attack_points >= 200 else BAD)

        # input_list[11]: Anthill logic
        input_list.append(BAD if our_anthill.captureHealth < 3 else GOOD)

        return input_list  # returns the list we created

    @staticmethod
    def score_state(state):
        """
        score_state: Compute a 'goodness' score of a given state for the current player.
        The score is computed by tallying up a total number of possible 'points',
        as well as a number of 'good' points.

        Various elements are weighted heavier than others, by providing more points.
        Some metrics, like food difference, is weighted by difference between the two
        players.

        Note: This is a staticmethod, it can be called without instancing this class.

        Parameters:
            state - GameState to score.
        """
        enemy_id = abs(state.whoseTurn - 1)
        our_inv = utils.getCurrPlayerInventory(state)
        enemy_inv = [
            inv for inv in state.inventories if inv.player == enemy_id].pop()
        we_win = 1.0
        enemy_win = 0.0
        our_food = our_inv.foodCount
        enemy_food = enemy_inv.foodCount
        food_difference = abs(our_food - enemy_food)
        our_anthill = our_inv.getAnthill()
        our_tunnel = our_inv.getTunnels()[0]
        enemy_anthill = enemy_inv.getAnthill()
        our_queen = our_inv.getQueen()
        enemy_queen = enemy_inv.getQueen()
        food_drop_offs = [our_tunnel.coords]
        food_drop_offs.append(our_anthill.coords)

        temp_list = [] * 12
        temp_list = AIPlayer.map_input(state)

        # Total points possible
        total_points = 1
        # Good points earned
        good_points = 0

        # Initial win condition checks:
        if (our_food == c.FOOD_GOAL or
                    enemy_queen is None or
                    enemy_anthill.captureHealth == 0):
            return we_win
        # Initial lose condition checks:
        if (enemy_food == c.FOOD_GOAL or
                    our_queen is None or
                    our_anthill.captureHealth == 0):
            return enemy_win

        # Score food
        total_points += (our_food + enemy_food) * 50  # 100
        good_points += our_food * 50  # 100

        # Differences over, say, 3 are weighted heavier
        if food_difference > 3:
            total_points += food_difference * 200  # 800
            if our_food > enemy_food:
                good_points += food_difference * 200  # 800

        # Carrying food is good
        food_move = 100
        our_workers = [ant for ant in our_inv.ants if ant.type == c.WORKER]

        # Food drop off points
        dropping_off = [
            ant for ant in our_workers if ant.coords in food_drop_offs and ant.carrying]

        # Depositing food is even better!
        if len(dropping_off) != 0:
            total_points += food_move * 30
            good_points += food_move * 30

        # Worker movement
        for ant in our_workers:
            ant_x = ant.coords[0]
            ant_y = ant.coords[1]
            for enemy in enemy_inv.ants:
                if ((abs(ant_x - enemy.coords[0]) > 3) and
                        (abs(ant_y - enemy.coords[1]) > 3)):
                    good_points += 60
                    total_points += 60
            if ant.carrying and ant not in dropping_off:
                # Good if carrying ants move toward a drop off.
                total_points += food_move
                good_points += food_move

                for dist in range(2, 4):
                    for dropoff in food_drop_offs:
                        if ((abs(ant_x - dropoff[0]) < dist) and
                                (abs(ant_y - dropoff[1]) < dist)):
                            good_points += food_move - (dist * 25)
                            total_points += food_move - (dist * 25)

        # Raw ant numbers comparison
        total_points += (len(our_inv.ants) + len(enemy_inv.ants)) * 10
        good_points += len(our_inv.ants) * 10

        # Weighted ant types
        # Workers, first 3 are worth 10, the rest are penalized
        enemy_workers = [ant for ant in enemy_inv.ants if ant.type == c.WORKER]
        if len(our_workers) <= 3:
            total_points += len(our_workers) * 10
            good_points += len(our_workers) * 10
        else:
            return 0.001
        total_points += len(enemy_workers) * 50

        # prefer workers to not leave home range
        our_range = [(x, y) for x in xrange(10) for y in xrange(5)]
        if len([ant for ant in our_workers if ant.coords not in our_range]) != 0:
            return .001

        # Offensive ants
        # Let's just say each ant is worth 20x its cost for now
        offensive = [c.SOLDIER, c.R_SOLDIER, c.DRONE]
        our_offense = [ant for ant in our_inv.ants if ant.type in offensive]
        enemy_offense = [
            ant for ant in enemy_inv.ants if ant.type in offensive]

        for ant in our_offense:
            ant_x = ant.coords[0]
            ant_y = ant.coords[1]
            attack_move = 160
            good_points += UNIT_STATS[ant.type][c.COST] * 20
            total_points += UNIT_STATS[ant.type][c.COST] * 20
            # good if on enemy anthill
            if ant.coords == enemy_anthill.coords:
                total_points += 100
                good_points += 100
            for enemy_ant in enemy_inv.ants:
                enemy_x = enemy_ant.coords[0]
                enemy_y = enemy_ant.coords[1]
                x_dist = abs(ant_x - enemy_x)
                y_dist = abs(ant_y - enemy_y)

                # good if attacker ant attacks
                if x_dist + y_dist == 1:
                    good_points += attack_move * 2
                    total_points += attack_move * 2

                # weighted more if closer to attacking
                for dist in xrange(1, 8):
                    if x_dist < dist and y_dist < dist:
                        good_points += attack_move - (dist * 20)
                        total_points += attack_move - (dist * 20)

        for ant in enemy_offense:
            total_points += UNIT_STATS[ant.type][c.COST] * 60

        # Stop building if we have more than 5 ants
        if len(our_inv.ants) > 5:
            total_points += 300

        # Queen stuff
        # Queen healths, big deal, each HP is worth 100!
        total_points += (our_queen.health + enemy_queen.health) * 100
        good_points += our_queen.health * 100
        queen_coords = our_queen.coords
        if queen_coords in food_drop_offs or queen_coords[1] > 2:
            # Stay off food_drop_offs and away from the front lines.
            # return .001
            total_points += 300

        # queen attacks if under threat
        for enemy_ant in enemy_inv.ants:
            enemy_x = enemy_ant.coords[0]
            enemy_y = enemy_ant.coords[1]
            x_dist = abs(queen_coords[0] - enemy_x)
            y_dist = abs(queen_coords[1] - enemy_y)

            if (x_dist + y_dist) == 1:
                good_points += 200
                total_points += 200

        # Anthill stuff
        total_points += (our_anthill.captureHealth +
                         enemy_anthill.captureHealth) * 200
        good_points += our_anthill.captureHealth * 200

        return float(good_points) / float(total_points)

    def evaluate_nodes(self, nodes):
        """Evalute a list of Nodes and returns the correct minmaxed score.
        That is, when it is our turn, return the max, when it is not, return
        the min."""
        # nodes = [node for node in nodes if node.score >= 0]
        if nodes[0].parent.state.whoseTurn == self.playerId:
            return max(nodes, key=lambda node: node.score)
        else:
            # print "Hey, a min node evaluation!"
            # print "Min: {}, Max: {}".format(min(nodes, key=lambda node:
            # node.score).score, max(nodes, key=lambda node: node.score).score)
            return min(nodes, key=lambda node: node.score)

    def get_best_move(self, curr_state, depth_limit, moves=None):
        """
        get_best_move: Returns the best move for a given state, searching to a given
        depth limit. Uses score_state() to find how 'good' a certain move is.

        The first depth level is done here, remaining levels are done in
        analyze_subnodes() recursively.

        Parameters:
            state - GameState to analyze
            depth_limit - Depth limit for search

        Returns:
            Move with the best score.
        """
        # Make a root pseudo-node
        root = Node(None, curr_state, -1)

        # If we get a list of moves, just get rid of the END move(s)
        if moves is None:
            all_moves = [move for move in utils.listAllLegalMoves(curr_state)]
        else:
            all_moves = [move for move in moves]

        # If there are moves left, then end the turn.
        if len(all_moves) == 1:
            return Move(c.END, None, None)
            # return Node(Move(c.END, None, None), state, 0.5)

        next_states = [AIUtils.getNextStateAdversarial(
            curr_state, move) for move in all_moves]

        # Build first level of nodes
        nodes = [Node(move, state, parent=root)
                 for move, state in zip(all_moves, next_states)]

        # nodes = [Node(move, None, -1, parent=root) for move in all_moves]

        # Analyze the subnodes for this state. nodes is modified in-place.
        best_node = self.analyze_subnodes(
            curr_state, depth_limit - 1, root, nodes)

        # If every move is bad, then just end the turn.
        if best_node.score <= 0.01:
            return Move(c.END, None, None)

        return best_node.move

    def analyze_subnodes(self, curr_state, depth_limit, root, nodes=None):
        """
        analyze_subnodes: This is the recursive method. Function stack beware.

        Analyze each subnode of a given state to a given depth limit.
        Update each node's score and return the highest-scoring subnode.

        Parameters:
            state - GameState to analyze
            depth_limit - Depth limit for search
            nodes (optional) - List of subnodes. Used if first depth
                level is computed elsewhere (in get_best_move)

        Returns:
            Best scoring analyzed sub-node.
        """
        # If nodes haven't been passed, then expand the current
        # state's subnodes.
        if nodes is None:
            all_moves = [move for move in utils.listAllLegalMoves(curr_state)]
            next_states = [AIUtils.getNextStateAdversarial(curr_state, move)
                           for move in all_moves]

            nodes = [Node(move, state, parent=root)
                     for move, state in zip(all_moves, next_states)]

            # nodes = [Node(move, None, -1, parent=root) for move in all_moves]

        # Prune the top or bottom 4/5 of nodes by score
        parent_max = root.state.whoseTurn == self.playerId
        nodes.sort(key=lambda node: node.score, reverse=parent_max)
        nodes = nodes[:int(math.ceil(len(nodes) ** (1. / 3.)))]

        # If the depth limit hasn't been reached,
        # analyze each subnode.
        if depth_limit >= 1:
            for node in nodes:

                if node.parent.alpha >= node.parent.beta:
                    # print "Prune at {}".format(node.parent)
                    break

                node.inherit_ab()
                # node.build_state()

                # Set the node's score to the best score of its subnodes.
                best_node = self.analyze_subnodes(
                    node.state, depth_limit - 1, node)
                node.score = best_node.score
                if node.parent.state.whoseTurn == self.playerId:  # Parent is a max node
                    if node.state.whoseTurn == self.playerId:
                        # We're a max node, so pass our beta up
                        node.parent.beta = min(node.beta, node.parent.beta)
                    else:
                        # We're a min node, so pass our beta up as alpha
                        node.parent.alpha = max(node.beta, node.parent.alpha)
                    # If we have a good move, the just use it.
                    if node.score > 0.7:
                        return node
                else:  # Parent is a min node
                    if node.state.whoseTurn == self.playerId:
                        # We're a max node, so pass our alpha up as beta
                        node.parent.beta = min(node.alpha, node.parent.beta)
                    else:
                        # We're a min node, so pass our alpha up
                        node.parent.alpha = max(node.alpha, node.parent.alpha)
                    # node.parent.beta = min(node.alpha, node.parent.beta)
                    if node.score < 0.3 and node.score >= 0:
                        return node


                        # If we have a good move, the just use it.
                        # if node.score > 0.7:
                        #     return node
        else:
            for node in nodes:
                if node.parent.alpha >= node.parent.beta:
                    # print "Prune at {}".format(node.parent)
                    break
                # node.inherit_ab()
                # node.build_state()
                node.calc_score()
                if node.parent.whose_turn() == self.playerId:  # Parent is max node
                    node.parent.alpha = max(node.score, node.parent.alpha)
                    if node.score > 0.7:
                        return node
                else:  # Parent is min node
                    node.parent.beta = min(node.score, node.parent.beta)
                    if node.score < 0.3:
                        return node
                        # print "Terminal node"

        # nodes = [node for node in nodes if node.score >= 0]
        # Prevent the ants form getting stuck when all moves
        # are equal.
        # random.shuffle(nodes)

        # Return the best node.
        # print "Examining " + str(len(nodes)) + " nodes"
        return self.evaluate_nodes(nodes)

    def getPlacement(self, currentState):
        """
        getPlacement:
            The getPlacement method corresponds to the
            action taken on setup phase 1 and setup phase 2 of the game.
            In setup phase 1, the AI player will be passed a copy of the
            state as current_state which contains the board, accessed via
            current_state.board. The player will then return a list of 11 tuple
            coordinates (from their side of the board) that represent Locations
            to place the anthill and 9 grass pieces. In setup phase 2, the
            player will again be passed the state and needs to return a list
            of 2 tuple coordinates (on their opponent's side of the board)
            which represent locations to place the food sources.
            This is all that is necessary to complete the setup phases.

        Parameters:
          current_state - The current state of the game at the time the Game is
              requesting a placement from the player.(GameState)

        Return: If setup phase 1: list of eleven 2-tuples of ints ->
                    [(x1,y1), (x2,y2),...,(x10,y10)]
                If setup phase 2: list of two 2-tuples of ints ->
                    [(x1,y1), (x2,y2)]
        """
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == c.SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw
                        # whatever I felt like in there.
                        currentState.board[x][y].constr is True
                moves.append(move)
            return moves
        elif currentState.phase == c.SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw
                        # whatever I felt like in there.
                        currentState.board[x][y].constr is True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    def getMove(self, currentState):
        """
        Description:
            Gets the next move from the Player.

        Parameters:
          current_state - The current state of the game at the time the Game is
              requesting a move from the player. (GameState)

        Return: Move(moveType [int],
                     coordList [list of 2-tuples of ints],
                     buildType [int])
        """

        depth = 3
        move = self.get_best_move(currentState, depth)

        return move

    def getAttack(self, currentState, attackingAnt, enemyLocations):
        """
        Description:
            Gets the attack to be made from the Player

        Parameters:
          current_state - The current state of the game at the time the
                Game is requesting a move from the player. (GameState)
          attackingAnt - A clone of the ant currently making the attack. (Ant)
          enemyLocation - A list of coordinate locations for valid attacks
            (i.e. enemies within range) ([list of 2-tuples of ints])

        Return: A coordinate that matches one of the entries of enemyLocations.
                ((int,int))
        """
        # Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]


class Node(object):
    """
    Simple class for a search tree Node.

    Each Node requires a Move and a GameState. If a score is not
    provided, then one is calculated with AIPlayer.score_state().
    """

    __slots__ = ('move', 'state', 'score', 'parent', 'alpha', 'beta')

    def __init__(self, move, state=None, score=None, parent=None):
        self.move = move
        self.state = state
        self.score = score
        self.alpha = -10  # Negative inf.
        self.beta = 10  # Positive inf.
        if score is None:
            self.score = AIPlayer.score_state(state)
        self.parent = parent
        if parent is not None:
            self.alpha = parent.alpha
            self.beta = parent.beta

    def calc_score(self):
        self.score = AIPlayer.score_state(self.state)

    def whose_turn(self):
        return self.state.whoseTurn

    def inherit_ab(self):
        if self.parent is not None:
            self.alpha = self.parent.alpha
            self.beta = self.parent.beta

    def build_state(self):
        self.state = AIUtils.getNextStateAdversarial(self.parent.state, self.move)


class AIUtils(object):
    @staticmethod
    def getNextState(currentState, move):
        """
        Version of genNextState with food carrying bug fixed.
        """
        # variables I will need
        myGameState = currentState.fastclone()
        myInv = utils.getCurrPlayerInventory(myGameState)
        me = myGameState.whoseTurn
        myAnts = myInv.ants

        # If enemy ant is on my anthill or tunnel update capture health
        myTunnels = myInv.getTunnels()
        myAntHill = myInv.getAnthill()
        for myTunnel in myTunnels:
            ant = utils.getAntAt(myGameState, myTunnel.coords)
            if ant is not None:
                opponentsAnts = myGameState.inventories[not me].ants
                if ant in opponentsAnts:
                    myTunnel.captureHealth -= 1
        if utils.getAntAt(myGameState, myAntHill.coords) is not None:
            ant = utils.getAntAt(myGameState, myAntHill.coords)
            opponentsAnts = myGameState.inventories[not me].ants
            if ant in opponentsAnts:
                myAntHill.captureHealth -= 1

        # If an ant is built update list of ants
        antTypes = [c.WORKER, c.DRONE, c.SOLDIER, c.R_SOLDIER]
        if move.moveType == c.BUILD:
            if move.buildType in antTypes:
                ant = Ant(myInv.getAnthill().coords, move.buildType, me)
                myInv.ants.append(ant)
                # Update food count depending on ant built
                if move.buildType == c.WORKER:
                    myInv.foodCount -= 1
                elif move.buildType == c.DRONE or move.buildType == c.R_SOLDIER:
                    myInv.foodCount -= 2
                elif move.buildType == c.SOLDIER:
                    myInv.foodCount -= 3

        # If a building is built update list of buildings and the update food
        # count
        if move.moveType == c.BUILD:
            if move.buildType == c.TUNNEL:
                building = Construction(move.coordList[0], move.buildType)
                myInv.constrs.append(building)
                myInv.foodCount -= 3

        # If an ant is moved update their coordinates and has moved
        if move.moveType == c.MOVE_ANT:
            newCoord = move.coordList[len(move.coordList) - 1]
            startingCoord = move.coordList[0]
            for ant in myAnts:
                if ant.coords == startingCoord:
                    ant.coords = newCoord
                    ant.hasMoved = False
                    # If an ant is carrying food and ends on the anthill or tunnel
                    # drop the food
                    if ant.carrying and ant.coords == myInv.getAnthill().coords:
                        myInv.foodCount += 1
                        # ant.carrying = False
                    for tunnels in myTunnels:
                        if ant.carrying and (ant.coords == tunnels.coords):
                            myInv.foodCount += 1
                            # ant.carrying = False
                    # If an ant doesn't have food and ends on the food grab
                    # food
                    if not ant.carrying:
                        foods = utils.getConstrList(
                            myGameState, None, (c.FOOD,))
                        for food in foods:
                            if food.coords == ant.coords:
                                ant.carrying = True
                    # If my ant is close to an enemy ant attack it
                    adjacentTiles = utils.listAdjacent(ant.coords)
                    for adj in adjacentTiles:
                        # If ant is adjacent my ant
                        if utils.getAntAt(myGameState, adj) is not None:
                            closeAnt = utils.getAntAt(myGameState, adj)
                            if closeAnt.player != me:  # if the ant is not me
                                closeAnt.health = closeAnt.health - \
                                                  UNIT_STATS[ant.type][c.ATTACK]  # attack
                                # If an enemy is attacked and looses all its health remove it from the other players
                                # inventory
                                if closeAnt.health <= 0:
                                    enemyAnts = myGameState.inventories[
                                        not me].ants
                                    for enemy in enemyAnts:
                                        if closeAnt.coords == enemy.coords:
                                            myGameState.inventories[
                                                not me].ants.remove(enemy)
                                # If attacked an ant already don't attack any
                                # more
                                break
        return myGameState

    @staticmethod
    def getNextStateAdversarial(currentState, move):
        """
        Description: This is the same as getNextState (above) except that it
        properly updates the hasMoved property on ants and the END move is
        processed correctly.

        Parameters:
          currentState - A clone of the current state (GameState)
          move - The move that the agent would take (Move)

        Return: A clone of what the state would look like if the move was made
        """
        # variables I will need
        nextState = AIUtils.getNextState(currentState, move)
        myInv = utils.getCurrPlayerInventory(nextState)
        myAnts = myInv.ants

        # If an ant is moved update their coordinates and has moved
        if move.moveType == c.MOVE_ANT:
            startingCoord = move.coordList[0]
            for ant in myAnts:
                if ant.coords == startingCoord:
                    ant.hasMoved = True
        elif move.moveType == c.END:
            for ant in myAnts:
                ant.hasMoved = False
            nextState.whoseTurn = 1 - currentState.whoseTurn
        return nextState
