import os
import torch
import gymnasium as gym
import numpy as np
from Utility_Functions import create_actor_distribution
from nn_builder.pytorch.NN import NN


class Tester(object):
    def __init__(self, config, agent_class, model_path="A3C.pt"):
        self.config = config
        default_hyperparameter_choices = {
            "output_activation": None,
            "hidden_activations": "relu",
            "dropout": 0.0,
            "initialiser": "default",
            "batch_norm": False,
            "columns_of_data_to_be_embedded": [],
            "embedding_dimensions": [],
            "y_range": (),
        }

        for key in default_hyperparameter_choices:
            if key not in config.hyperparameters["Actor_Critic_Agents"].keys():
                config.hyperparameters["Actor_Critic_Agents"][key] = (
                    default_hyperparameter_choices[key]
                )
        self.hyperparameters = config.hyperparameters["Actor_Critic_Agents"]

        self.env = gym.make(self.config.environment_name, render_mode="human")
        self.action_types = (
            "DISCRETE" if self.env.action_space.dtype == np.int64 else "CONTINUOUS"
        )
        self.state_size = self.get_state_size()
        self.action_size = int(self.get_action_size())
        self.actor_critic = NN(
            input_dim=self.state_size,
            layers_info=self.hyperparameters["linear_hidden_units"]
            + [[self.action_size, 1]],
            output_activation=self.hyperparameters["final_layer_activation"],
            batch_norm=self.hyperparameters["batch_norm"],
            dropout=self.hyperparameters["dropout"],
            hidden_activations=self.hyperparameters["hidden_activations"],
            initialiser=self.hyperparameters["initialiser"],
            columns_of_data_to_be_embedded=self.hyperparameters[
                "columns_of_data_to_be_embedded"
            ],
            embedding_dimensions=self.hyperparameters["embedding_dimensions"],
            y_range=self.hyperparameters["y_range"],
        ).to("cuda:0")
        self.load_agent(agent_class[0], model_path)
        self.episode_scores = []

    def load_agent(self, agent_class, model_path):
        """创建Agent并加载训练好的模型"""

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = "cpu"

        if os.path.exists(model_path):
            self.actor_critic.load_state_dict(
                torch.load(model_path, map_location=map_location)
            )
            print(f"成功加载模型: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    def run_test_episodes(self, num_episodes=10, render=True):
        """运行指定次数的测试回合"""
        print(f"\n开始测试({num_episodes}个回合)...")

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                self.env.render()

                action = self.choose_action(state)
                next_state, reward, done, truncate, _ = self.env.step(action)

                total_reward += reward
                state = next_state
                steps += 1
                done = done or truncate

            self.episode_scores.append(total_reward)
            print(
                f"回合 {ep+1}/{num_episodes} | 总奖励: {total_reward:.1f} | 步数: {steps}"
            )

        self._show_summary_stats()

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).cuda()
        model_output = self.actor_critic(state)
        actor_output = model_output[:, : self.action_size]
        action_distribution = create_actor_distribution(
            self.action_types, actor_output, self.action_size
        )
        action = action_distribution.sample().cpu().numpy()
        if self.action_types == "DISCRETE":
            action = action.item()
        else:
            action = action[0]
        return action

    def _show_summary_stats(self):
        """显示测试结果统计"""
        avg_score = np.mean(self.episode_scores)
        std_score = np.std(self.episode_scores)
        max_score = np.max(self.episode_scores)
        min_score = np.min(self.episode_scores)

        print("\n=== 测试结果 ===")
        print(f"平均奖励: {avg_score:.2f} ± {std_score:.2f}")
        print(f"最大奖励: {max_score:.2f} | 最小奖励: {min_score:.2f}")
        print(f"测试回合数: {len(self.episode_scores)}")

    def load_model(self, path_to_model="A3C.pt"):
        """Loads a model from a given path"""
        self.agents.load_model(path_to_model)

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "overwrite_action_size" in self.config.__dict__:
            return self.config.overwrite_action_size
        if "action_size" in self.env.__dict__:
            return self.env.action_size
        if self.action_types == "DISCRETE":
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        random_state, _ = self.env.reset()
        if isinstance(random_state, dict):
            state_size = (
                random_state["observation"].shape[0]
                + random_state["desired_goal"].shape[0]
            )
            return state_size
        else:
            return random_state.size
