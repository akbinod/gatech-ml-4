import numpy as np
import random
import Agent


def loss_converged():
	'''The last 10 mean loss values must be within theta of each other for this to return true.'''
	ct = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	ll = len(ct)
	means = []
	if ll > 10:
		# if the average loss of the last 10 loss scores
		# is within theta of each other, then we're converged
		for i in range(10):
			means.append(np.mean(ct[0 :ll-i]))
		print(means)
		done = True
		for i in range(len(means)-1):
			if abs(means[i] - means[i+1]) > .001:
				done = False




def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


loss_converged()
# # s = (1,2,3, (1,1))
# # a = np.array(s)
# # a = np.zeros((2,2), dtype=float)
# # b = np.reshape(a,(1,4))
# # b = np.expand_dims(b,1)
# # print(b.shape)
# # print(b)

# # b = a[:, np.newaxis]
# # b = np.expand_dims(a,4)
# # print(b.shape)
# # b[0][0][0][0] = (1,2)
# # print(b[0])
# r = c = 2
# a = np.zeros((3, (r*c)),dtype=float)
# a[0][0] = a[0][1] = a[0][2] = a[0][3] = 2
# a[1][0] = a[1][1] = a[1][2] = a[1][3] = 4
# a[2][0] = a[2][1] = a[2][2] = a[2][3] = 8

# mul = np.array([0.8,0.1,0.1])

# b = mul.dot(a)
# b = np.reshape(b, (r,c))
# print(b)
# width = 2
# l = random.choices(population=np.arange(width),k=width*width)
# print(l)
# proposed_policy = np.array(l)
# proposed_policy = np.reshape(proposed_policy, (width, width))
# print(proposed_policy)

print(generate_random_map(size=64))

print(str(Agent.convergence_measure.not_yet))