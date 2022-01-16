import numpy as np

class Node():
    _VAL_WIN_X = 1
    _VAL_WIN_Y = -1
    _VAL_DRAW = 0

    def __init__(self, parent, state):
        self._parent = parent
        self._children = {}
        self._state = state
        self._leaf = False
        self._value = self._get_value()

    def add_child(self, node, action):
        self._children[action] = node

    def _get_value(self):
        if self._check_winner(1):
            self._leaf = True
            return self._VAL_WIN_X

        if self._check_winner(2):
            self._leaf = True
            return self._VAL_WIN_Y

        if 0 not in self._state:
            self._leaf = True
            return self._VAL_DRAW

        return 0


    def _check_winner(self, player):
        # horizontal
        for i in range(0, 9, 3):
            if self._state[i] == self._state[i+1] == self._state[i+2] == player:
                return True

        # vertical
        for i in range(3):
            if self._state[i] == self._state[i+3] == self._state[i+6] == player:
                return True

        # diagonal ppal
        if self._state[0] == self._state[4] == self._state[8] == player:
            return True

        # diagonal sec
        if self._state[2] == self._state[4] == self._state[6] == player:
            return True

    def __str__(self):
        symbol = [' ', 'X', 'O']
        data = [symbol[i] for i in self._state]
        res = f' {data[0]} | {data[1]} | {data[2]}'
        res += '\n――― ――― ―――'
        res += f'\n {data[3]} | {data[4]} | {data[5]}'
        res += '\n――― ――― ―――'
        res += f'\n {data[6]} | {data[7]} | {data[8]}'
        return res

class Tree():
    def __init__(self, state):
        self._root = Node(None, state)
        self._n = 1     # num nodos
        self._populate()

    def _populate(self):
        node = self._root
        q = []  # cola de nodos a procesar
        if not node._leaf:   # añadimos el raíz
            q.insert(0, node)

        while len(q):   # mientras haya nodos en la cola
            parent = q.pop() # siguiente nodo a procesar
            parent_st = np.array(parent._state) # estado del nodo padre
            count_X = np.where(parent_st==1)[0].size
            count_O = np.where(parent_st==2)[0].size
            player = 1 if count_X == count_O else 2 # siguiente jugador?

            # creamos un nodo hijo por cada acción posible
            for action in np.where(parent_st==0)[0]:
                child_st = np.array(parent_st)
                child_st[action] = player
                child_node = Node(parent, child_st)
                parent.add_child(child_node, action)
                self._n += 1
                if not child_node._leaf:   # añadimos el nuevo nodo a la cola
                    q.insert(0, child_node)




