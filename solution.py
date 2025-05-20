import math
import random
from collections import deque
from copy import deepcopy


class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:
        self.phoneme_table = phoneme_table
        self.phoneme_table_map = {}
        for phoneme, replacements in phoneme_table.items():
            for replacement in replacements:
                if replacement not in self.phoneme_table_map:
                    self.phoneme_table_map[replacement] = []
                self.phoneme_table_map[replacement].append(phoneme)
        # print("Inverted Phoneme Map", self.phoneme_table_map)
        self.vocabulary = vocabulary
        self.best_state = None
        self.best_visited=None
        self.tabu_list = deque(maxlen=25)
        self.k = 5  # Number of neighbors to consider
        self.replaced_indices = {}  # Store the indices of the replaced characters

    def asr_corrector(self, environment):
        """
        Correct the ASR errors in the provided initial state by using simulated annealing with informed search.
        """
        # Initialize the current state and its cost
        self.best_state = environment.init_state
        current_state = environment.init_state
        best_cost = current_cost = environment.compute_cost(current_state)
        visited=[]
        is_insertion=[False,False]
        for j in range (0,len(current_state)):
            if current_state[j]==" ":
                visited.append(1)
            else:
                visited.append(0)

        print(f"Initial state: {current_state}; Visited: {visited}; Cost: {current_cost}")

        # Simulated Annealing parameters ~ 43 iterations
        T = 0.1  # Initial temperature
        T_min = 0.0001  # Minimum temperature
        alpha = 0.85  # Cooling rate

        iteration = 0
        accepted = 0  # Track the number of accepted neighbors

        while T > T_min:
            if iteration < 6:
                self.k = 5
            elif iteration < 12:
                self.k = 4
            elif iteration < 18:
                self.k = 3
            elif iteration < 24:
                self.k = 2
            else:
                self.k = 1
            iteration += 1

            print(
                f"Iteration {iteration}; Current state: {current_state}; Visited: {visited}; Cost: {current_cost}; k: {self.k}"
            )
            # Generate a neighbor state with a heuristic-based approach
            neighbor_states, neighbor_costs, neighbor_visited,neighbor_inserted = self.generate_informed_neighbors(
                current_state, environment, visited, is_insertion,iteration
            )
            print(f"Top ${self.k} Neighbor states: {neighbor_states}")
            print(f"Top ${self.k} Neighbor costs: {neighbor_costs}")

            if not len(neighbor_states):
                if iteration<=2:
                    continue
                print("No neighbors found. Terminating search.")
                break

            # Choose a random neighbor state
            random_index = random.randint(0, len(neighbor_states) - 1)
            neighbor_state = neighbor_states[random_index]
            neighbor_visit= neighbor_visited[random_index]
            neighbor_is_insertion=neighbor_inserted[random_index]
            neighbor_cost_dict = dict(zip(neighbor_states, neighbor_costs))
            neighbor_cost = neighbor_cost_dict.get(neighbor_state, None)
            print(f"Chosen neighbor state: {neighbor_state} with cost: {neighbor_cost}")

            # Calculate the change in cost
            delta_cost = neighbor_cost - current_cost

            # Decide whether to accept the neighbor state
            # if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / T):
            # Check if the first or last word insertion has been made
            current_words = current_state.split()
            neighbor_words = neighbor_state.split()

            # Accept the neighbor state
            current_state = neighbor_state
            visited=neighbor_visit
            current_cost = neighbor_cost
            is_insertion=neighbor_is_insertion
            accepted += 1

            # Update Tabu list with the current state
            self.tabu_list.append(current_state)

            # If this state is the best we've seen, update the best state
            if neighbor_cost < best_cost:
                self.best_state = neighbor_state
                self.best_visited=neighbor_visit
                best_cost = neighbor_cost
                print(f"Iteration {iteration}: New best state with cost {best_cost}")
            # else:
            #     print("Neighbor state rejected")

            # Cool down the temperature
            T *= alpha

        # Save the final best state
        print(f"Found best state ${self.best_state} with cost {best_cost}")

    def generate_informed_neighbors(self, state, environment, visited,is_insertion,iteration):
        # print("Generating neighbors for state:", state)
        neighbors = []
        if iteration>2:
            neighbors.extend(self.generate_neighbors_by_substitution(state, visited,is_insertion))
        else:
            neighbors.extend(self.generate_neighbors_by_insertion(state, visited,is_insertion))

        if neighbors:
            # Filter out neighbors that are in the Tabu list
            valid_neighbors = [s for s in neighbors if s[0] not in self.tabu_list]
            print(
                f"Found {len(valid_neighbors)} valid neighbors out of {len(neighbors)}"
            )

            if not valid_neighbors:
                print("No valid neighbors found. Implementing fallback mechanism.")
                valid_neighbors = neighbors  # Fallback mechanism - need to improve

            neighbors_with_costs = [
                (s[0],s[1], environment.compute_cost(s[0]),s[2]) for s in valid_neighbors
            ]
            sorted_neighbors_with_costs = sorted(
                neighbors_with_costs, key=lambda x: x[2]
            )
            top_k_neighbors = [s for s,visited,cost,insertion in sorted_neighbors_with_costs[:self.k]]
            top_k_costs = [cost for s,visited,cost,insertion in sorted_neighbors_with_costs[:self.k]]
            top_k_visited = [visited for s,visited,cost,insertion in sorted_neighbors_with_costs[:self.k]]
            top_k_insertions=[insertion for s,visited,cost,insertion in sorted_neighbors_with_costs[:self.k]]

            return top_k_neighbors, top_k_costs, top_k_visited,top_k_insertions
        else:
            print("No neighbors found.")
            return [], [], [],[]

    def generate_neighbors_by_substitution(self, state, visited,is_insertion):
            # print("Generating neighbors by substitution for state:", state)
            neighbors = []
            words = list(state)
            for j, char in enumerate(words):
                if visited[j]==1:
                    continue
                if char in self.phoneme_table_map:
                    for replacement in self.phoneme_table_map[char]:
                        if replacement != char:
                            new_words=deepcopy(words)
                            new_words[j]=replacement
                            copy=deepcopy(visited)
                            copy[j]=1
                            if len(replacement)==2:
                                copy.insert(j+1,1)
                            new="".join(new_words)
                            neighbors.append([new,copy,is_insertion])
                if j < len(words) - 1:
                    if visited[j+1]==True:
                        continue
                    bigram = words[j]+words[j+1]
                    if bigram in self.phoneme_table_map:
                        for replacement in self.phoneme_table_map[bigram]:
                            if replacement != bigram:
                                new_words=deepcopy(words)
                                copy=deepcopy(visited)
                                if len(replacement)==1:
                                    new_words[j]=replacement
                                    new_words.pop(j+1)
                                    copy[j]=1
                                    copy.pop(j+1)
                                    new="".join(new_words)
                                else:
                                    new_words[j]=bigram[0]
                                    new_words[j+1]=bigram[1]
                                    copy[j]=1
                                    copy[j+1]=1
                                    new="".join(new_words)
                                neighbors.append([new,copy,is_insertion])
            return neighbors
    def generate_neighbors_by_insertion(self, state,visited,is_insertion):
        # print("Generating neighbors by insertion for state:", state)
        neighbors = []

        
        if (is_insertion[0]==True and is_insertion[1]==True):
          # If both first and last word insertions have been made
            return neighbors
        
        copy_insertion=deepcopy(is_insertion)

        for word in self.vocabulary:
            if is_insertion[0]==False:
                v_copy=deepcopy(visited)
                v_copy = [1] * (len(word) + 1) + v_copy
                copy_insertion[0]=True
                neighbors.append([word + " " + state,v_copy,copy_insertion])

            if is_insertion[1]==False:
                v_copy=deepcopy(visited)
                v_copy = v_copy + [1] * (len(word) + 1) 
                copy_insertion[1]=True

                neighbors.append([state + " " + word,v_copy,copy_insertion])
                
        return neighbors