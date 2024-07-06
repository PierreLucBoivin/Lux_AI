import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


# inspired from https://github.com/lubiluk/ddpg
class DDPG:
    def __init__(self, config):

        self.state_dim = config['state_dim']
        print("Size of State Space ->  {}".format(self.state_dim))
        self.state_n = config['state_n']

        self.subgoal_dim = config['subgoal_dim']
        print("Size of Subgoal Space ->  {}".format(self.subgoal_dim))
        self.subgoal_n = config['subgoal_n']

        self.std_dev = config['std_dev'] # 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = config['critic_lr'] # 0.002
        self.actor_lr = config['actor_lr'] #0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Discount factor for future rewards
        self.gamma = config['gamma']
        # Used to update target networks
        self.tau = config['tau']

        self.buffer = BufferH(self, self.state_dim, self.subgoal_n, 50000, 64)

    def policy(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        noise_object = self.ou_noise
        prob = tf.squeeze(self.actor_model(state))
        # noise = noise_object()
        # Adding noise to action
        # sampled_subgoal = sampled_subgoal.numpy() + noise
        prob = prob.numpy()
        #print(prob)
        # prob += noise_object()

        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        subgoal = dist.sample()
        # We make sure action is within bounds
        # legal_subgoal = np.clip(sampled_subgoal, self.lower_bound, self.upper_bound)

        return subgoal.numpy()

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim,))
        out = layers.Reshape((1, self.state_dim,))(inputs)
        out = layers.LSTM(4)(out)
        out = layers.Dense(256, activation="relu")(out)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.subgoal_n, activation="softmax", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        # outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.state_dim))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.subgoal_n))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def learn(self):
        self.buffer.learn()
        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def record(self, state, subgoal, reward, prev_state):
        self.buffer.record((prev_state, subgoal, reward, state))


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class BufferH:
    def __init__(self, ddpg: DDPG, num_states, num_subgoal, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        self.ddpg = ddpg

        self.num_states = num_states
        self.num_subgoal = num_subgoal

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.subgoal_buffer = np.zeros((self.buffer_capacity, num_subgoal))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def clear(self):
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.subgoal_buffer = np.zeros((self.buffer_capacity, self.num_subgoal))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

    # Takes (s,g,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.subgoal_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, state_batch, subgoal_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.ddpg.target_actor(next_state_batch, training=True)
            y = reward_batch + self.ddpg.gamma * self.ddpg.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.ddpg.critic_model([state_batch, subgoal_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.ddpg.critic_model.trainable_variables)
        self.ddpg.critic_optimizer.apply_gradients(
            zip(critic_grad, self.ddpg.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.ddpg.actor_model(state_batch, training=True)
            critic_value = self.ddpg.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.ddpg.actor_model.trainable_variables)
        self.ddpg.actor_optimizer.apply_gradients(
            zip(actor_grad, self.ddpg.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.subgoal_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


if __name__ == '__main__':
    problem = 'Taxi-v3'
    env = gym.make(problem)
    config = {
        'state_dim':1, # env.observation_space.shape[0],
        'state_n' : env.observation_space.n,
        'subgoal_dim': 1, #env.action_space.shape[0],
        'subgoal_n' : env.action_space.n,
        'std_dev':0.2,
        'critic_lr': 0.002,
        'actor_lr': 0.001,
        'gamma' : 0.99,
        'tau': 0.005,
    }
    ddpg = DDPG(config)
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    total_episodes = 1000

    # Takes about 4 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            # env.render()

            subgoal = ddpg.policy(prev_state)
            # Recieve state and reward from environment.
            state, reward, done, info = env.step(subgoal)
            # print(state, reward,done)
            ddpg.record(state, subgoal, reward, prev_state)
            episodic_reward += reward



            # End this episode when `done` is True
            if done:
                break

            prev_state = state
        ddpg.learn()
        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
