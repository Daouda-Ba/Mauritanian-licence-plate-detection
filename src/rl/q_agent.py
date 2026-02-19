import random
from collections import defaultdict

class QAgent:
    def __init__(self, alpha=0.2, gamma=0.95, eps=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q = defaultdict(lambda: [0.0, 0.0])  # actions: 0 wait, 1 OCR

    def act(self, state):
        if random.random() < self.eps:
            return random.choice([0, 1])
        q0, q1 = self.Q[state]
        return 0 if q0 >= q1 else 1

    def update(self, s, a, r, s_next):
        best_next = max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s][a]
        self.Q[s][a] += self.alpha * td_error
        
    def q_table_size(self):
        return len(self.Q)