import random
import matplotlib.pyplot as plt

# A normally distributed random variable with expected value 0 and variance 1

# for _ in range(10):
# 	if(random.random() < .5):
# 		print "head"
# 	else:
# 		print "tail"

class Coin(object):
	sides = ('heads', 'tails')
	last_result = None

	def flip(self):
		self.last_result = result = random.choice(self.sides)
		return result

def create_coins(number):
	return [Coin() for _ in xrange(number)]

def flip_coins(coins):
	for coin in coins:
		coin.flip()

def count_heads(flipped_coins):
	return sum(coin.last_result == 'heads' for coin in flipped_coins)

def count_tails(flipped_coins):
	return sum(coin.last_result == 'tails' for coin in flipped_coins)

def main():
	coins = create_coins(1000)
	test = []
	for i in xrange(100):
		flip_coins(coins)
		test.append(count_heads(coins))
	print sum(test)/len(test)
	plt.hist(test)
	plt.show()


if __name__ == '__main__':
	main()

### Normal variable. Mean is 0, variance is 1

from numpy.random import normal
s = normal(size=(1024*32,))
hist(s, bins=50)

for _ in range(10):
	print np.var(normal(size=(1024*32,)))