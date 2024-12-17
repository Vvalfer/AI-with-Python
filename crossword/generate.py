import sys

from crossword import *
from collections import deque


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """Rendre chaque variable cohérente avec les nœuds (longueur du mot)."""
        for var in self.domains:
            to_remove = {word for word in self.domains[var] if len(word) != var.length}
            self.domains[var] -= to_remove

    def revise(self, x, y):
        """Rendre la variable `x` cohérente avec la variable `y`."""
        revised = False
        i, j = self.crossword.overlaps[x, y]
        
        if i is not None and j is not None:
            to_remove = set()
            for x_word in self.domains[x]:
                if not any(x_word[i] == y_word[j] for y_word in self.domains[y]):
                    to_remove.add(x_word)
            if to_remove:
                self.domains[x] -= to_remove
                revised = True
        return revised

    def ac3(self, arcs=None):
        """Appliquer l'algorithme AC3 pour garantir la cohérence des arcs."""
        queue = deque(arcs) if arcs else deque([(x, y) for x in self.domains for y in self.crossword.neighbors(x) if x != y])
        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """Vérifie si toutes les variables ont une affectation."""
        return all(var in assignment for var in self.crossword.variables)

    def consistent(self, assignment):
        """Vérifie si l'affectation est cohérente (respect des longueurs, unicité, etc.)."""
        words = set()
        for var, word in assignment.items():
            # Vérifie la longueur
            if len(word) != var.length:
                return False
            # Vérifie l'unicité
            if word in words:
                return False
            words.add(word)
            # Vérifie les conflits avec les voisins
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if i is not None and j is not None and assignment[var][i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """Retourne les valeurs du domaine triées par "valeur la moins contraignante"."""
        def constraint_count(value):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    count += sum(1 for neighbor_word in self.domains[neighbor] if neighbor_word[j] != value[i])
            return count

        return sorted(self.domains[var], key=constraint_count)


    def select_unassigned_variable(self, assignment):
        """Sélectionne la variable non affectée selon la règle MRV puis le degré de contrainte."""
        unassigned_vars = [var for var in self.crossword.variables if var not in assignment]
        return min(unassigned_vars, key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var))))


    def backtrack(self, assignment):
        """Utilise la recherche par retour arrière pour trouver une solution complète."""
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result:
                    return result
        return None



def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
