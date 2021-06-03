import neptune

# The init() function called this way assumes that
# NEPTUNE_API_TOKEN environment variable is defined.


def main():
  for i in range(100):
    neptune.log_metric('loss', 0.95**i)

  neptune.log_metric('AUC', 0.96)


if __name__ == '__main__':
  neptune.init('ethangoan/sandbox')
  neptune.create_experiment(name='minimal_example')
  main()
# log some metrics
