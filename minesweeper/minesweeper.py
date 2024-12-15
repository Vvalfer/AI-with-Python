import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """
    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # Si toutes les cellules sont des mines
        if len(self.cells) == self.count:
            return self.cells.copy()
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # Si aucune cellule n'est une mine
        if self.count == 0:
            return self.cells.copy()
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    def __init__(self, height=8, width=8):

        # Taille de la grille
        self.height = height
        self.width = width

        # Suivi des coups joués, des mines connues et des cases sûres
        self.moves_made = set()
        self.mines = set()
        self.safes = set()

        # Connaissances sur le jeu (phrases logiques)
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marque une cellule comme étant une mine et met à jour les phrases logiques.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marque une cellule comme étant sûre et met à jour les phrases logiques.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Met à jour la base de connaissances de l'IA.
        """
        # 1️⃣ Marquer la cellule comme jouée et sûre
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # 2️⃣ Trouver les cellules voisines de la cellule
        neighbors = set()
        i, j = cell
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if (di, dj) != (0, 0):  # Ne pas inclure la cellule elle-même
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.height and 0 <= nj < self.width:  # Limites de la grille
                        if (ni, nj) not in self.safes and (ni, nj) not in self.mines:
                            neighbors.add((ni, nj))

        # 3️⃣ Ajouter la nouvelle phrase à la connaissance
        if neighbors:
            new_sentence = Sentence(neighbors, count)
            self.knowledge.append(new_sentence)

        # 4️⃣ Marquer les nouvelles mines et cases sûres
        self.update_knowledge()

    def update_knowledge(self):
        """
        Identifie les nouvelles mines et cases sûres à partir de la connaissance.
        """
        all_safes = set()
        all_mines = set()

        for sentence in self.knowledge:
            all_safes.update(sentence.known_safes())
            all_mines.update(sentence.known_mines())

        for cell in all_safes:
            self.mark_safe(cell)
        for cell in all_mines:
            self.mark_mine(cell)

        # Supprimer les phrases vides
        self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

        # 5️⃣ Inférer de nouvelles phrases en utilisant la méthode de "sous-ensemble"
        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                if sentence1 != sentence2 and sentence1.cells.issubset(sentence2.cells):
                    new_cells = sentence2.cells - sentence1.cells
                    new_count = sentence2.count - sentence1.count
                    new_sentence = Sentence(new_cells, new_count)
                    if new_sentence not in self.knowledge:
                        self.knowledge.append(new_sentence)

    def make_safe_move(self):
        """
        Renvoie un mouvement sûr qui n'a pas encore été effectué.
        """
        possible_moves = self.safes - self.moves_made
        if possible_moves:
            return possible_moves.pop()
        return None

    def make_random_move(self):
        """
        Renvoie un mouvement aléatoire sur le plateau.
        """
        possible_moves = set(
            (i, j) for i in range(self.height) for j in range(self.width)
        ) - self.moves_made - self.mines
        if possible_moves:
            return random.choice(list(possible_moves))
        return None