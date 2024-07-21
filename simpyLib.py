import simpy

# Define a simple process
def car(env):
    while True:
        print(f'Start parking at {env.now}')
        yield env.timeout(5)  # Wait for 5 time units

        print(f'Start driving at {env.now}')
        yield env.timeout(2)  # Wait for 2 time units

# Create an environment
env = simpy.Environment()

# Add the process to the environment
env.process(car(env))

# Run the simulation for 15 time units
env.run(until=15)
