import random
from collections import Counter, deque
import gym
import numpy as np
from gym.spaces import Box, Discrete, Space
from src.landmarks.common import (
    extract_state_from_image,
    PANDAS_FCSV_KWARGS,
)
from src.landmarks.transforms import LoadCSVd, TransformLandmarksd, ExtractLandmarks2d
from monai.transforms import Compose
from copy import deepcopy
from pathlib import Path


class MultiAgentLandmarkEnv(gym.Env):
    """
    Actions
    """

    X_PLUS = 0
    X_MINUS = 1
    Y_PLUS = 2
    Y_MINUS = 3
    Z_PLUS = 4
    Z_MINUS = 5

    def __init__(
        self,
        landmarks: list = ["AC", "PC", "VN4"],
        max_steps: int = 500,
        state_size: list = [25, 25, 25],
        clip_reward: int = 1,
        image_tag: str = "og",
        mode: str = "train",
        history_length: int = 4,
        extensions: dict = {},
        **kwargs,
    ):
        self.landmark_names = [landmark_name.upper() for landmark_name in landmarks]
        self.max_steps = max_steps
        self.spatial_state_size = state_size
        self.clip_reward = clip_reward
        self.image_tag = image_tag
        self.mode = mode
        self.history_length = history_length

        self.n_agents = len(self.landmark_names)
        self.stage = extensions["stage"]

        # multi resolution extension
        if extensions["multi_res"]["use"]:
            self.resolutions = extensions["multi_res"]["resolutions"]
        else:
            self.resolutions = extensions["base_resolution"]

        # map resolutions to action_sizes
        self.action_sizes = [res * res if res > 1 else res for res in self.resolutions]
        print(self.action_sizes)

        # noisy location initialization extension
        if extensions["noisy_initialization"]["use"]:
            self.noisy_initialization = extensions["noisy_initialization"]["offset"]
        else:
            self.noisy_initialization = 0

        self.location_vector_size = (len(self.landmark_names) - 1) * 3

        # priors initialization extension
        self.location_priors = None
        self.use_priors = extensions["priors_initialization"]["use"]
        if self.use_priors:
            # self.priors = extensions["priors_initialization"]["priors"]
            self.priors = (
                Path(__file__).parent.resolve() / "train_priors.fcsv"
            ).as_posix()
        self.dataset = []
        self.dataset_iter = iter(self.dataset)
        # define action and state space
        self.action_space = Discrete(6)
        self.observation_space = Box(
            low=0, high=255, shape=self.spatial_state_size, dtype=np.uint8
        )

        # environment state variables
        # fill this when reset() is called
        self.episode_data = None
        self.goal_locations = None
        self.step_count = None
        self.agent_location_history_physical = None
        self.agent_location_history_voxel = None
        self.q_value_history = None
        self.resolution_index = None
        self.obs_history = None

    def __oscillating(self, agent_id):
        counter = Counter(self.agent_location_history_voxel[agent_id])

        most_common = counter.most_common()
        if most_common[0][0] == (0, 0, 0):
            if most_common[1][1] > 3:
                # log.info(f"OSC: ({most_common[:4]})")
                return True
        elif most_common[0][1] > 3:
            # log.info(f"OSC: ({most_common[:4]})")
            return True

        return False

    def __inbounds(self, agent_id):
        current_lps_position = self.agent_location_history_physical[agent_id][-1]

        continuous_voxel_space_position = self.episode_data[
            "image"
        ].TransformPhysicalPointToContinuousIndex(current_lps_position)

        image_size = np.array(
            self.episode_data["image"].GetLargestPossibleRegion().GetSize()
        )

        for axis_index in range(len(image_size)):
            if (
                continuous_voxel_space_position[axis_index] < 0
                or continuous_voxel_space_position[axis_index] > image_size[axis_index]
            ):
                # log.info(
                #     f"Agent({agent_id}) out of bounds || loc({continuous_voxel_space_position}) || bounds({image_size})"
                # )
                return False

        return True

    def stop_condition_helper(self, true_location, agent_location, resolution):
        diff = np.absolute(np.subtract(true_location, agent_location))
        if diff.max() <= 0.5 * resolution:
            return True
        return False

    def __done(self, agent_id) -> bool:
        # step condition
        if self.step_count > self.max_steps:
            return True

        # oscillation condition
        if (
            self.__oscillating(agent_id)
            and self.resolution_index[agent_id] == len(self.resolutions) - 1
        ):
            return True

        if self.mode == "train":
            if self.resolution_index[agent_id] == len(
                self.resolutions
            ) - 1 and self.stop_condition_helper(
                self.goal_locations[agent_id],
                self.agent_location_history_physical[agent_id][-1],
                self.resolutions[self.resolution_index[agent_id]],
            ):
                dist = np.linalg.norm(
                    self.goal_locations[agent_id]
                    - self.agent_location_history_physical[agent_id][-1]
                )
                log.info(
                    f"Agent({agent_id}) found landmark {dist:.3f} < {self.resolutions[self.resolution_index[agent_id]]}"
                )
                return True

        return False

    def __reward(self, agent_id) -> float:
        current_location = self.agent_location_history_physical[agent_id][-1]
        current_dist_to_lmk = np.linalg.norm(
            np.array(current_location) - np.array(self.goal_locations[agent_id])
        )

        previous_location = self.agent_location_history_physical[agent_id][-2]
        previous_dist_to_lmk = np.linalg.norm(
            np.array(previous_location) - np.array(self.goal_locations[agent_id])
        )

        return previous_dist_to_lmk - current_dist_to_lmk

    def __observation(self, agent_id) -> Space:
        # generate observations for all agents that are not done
        obs = extract_state_from_image(
            self.episode_data["image"],
            self.spatial_state_size,
            self.agent_location_history_physical[agent_id][-1],
            resolution=self.resolutions[self.resolution_index[agent_id]],
        ).astype(np.uint8)

        assert self.observation_space.contains(
            obs
        ), "MultiAgentLandmarkEnv returned invalid observation"

        if agent_id not in self.obs_history.keys():
            self.obs_history[agent_id] = deque(maxlen=self.history_length)
            # fill obs history
            for _ in range(self.history_length):
                self.obs_history[agent_id].append(obs)

        else:
            self.obs_history[agent_id].append(obs)

        return list(self.obs_history[agent_id])

    def __apply_action(self, agent_id, action):
        lps_cord = self.agent_location_history_physical[agent_id][-1]
        new_lps_cord = None
        voxel_cord = self.agent_location_history_voxel[agent_id][-1]
        new_voxel_cord = None
        action_size = self.action_sizes[self.resolution_index[agent_id]]

        if action == self.X_PLUS:  # 0
            new_lps_cord = [lps_cord[0] + action_size, lps_cord[1], lps_cord[2]]
            new_voxel_cord = (voxel_cord[0] + 1, voxel_cord[1], voxel_cord[2])

        elif action == self.X_MINUS:  # 1
            new_lps_cord = [lps_cord[0] - action_size, lps_cord[1], lps_cord[2]]
            new_voxel_cord = (voxel_cord[0] - 1, voxel_cord[1], voxel_cord[2])

        elif action == self.Y_PLUS:  # 2
            new_lps_cord = [lps_cord[0], lps_cord[1] + action_size, lps_cord[2]]
            new_voxel_cord = (voxel_cord[0], voxel_cord[1] + 1, voxel_cord[2])

        elif action == self.Y_MINUS:  # 3
            new_lps_cord = [lps_cord[0], lps_cord[1] - action_size, lps_cord[2]]
            new_voxel_cord = (voxel_cord[0], voxel_cord[1] - 1, voxel_cord[2])

        elif action == self.Z_PLUS:  # 4
            new_lps_cord = [lps_cord[0], lps_cord[1], lps_cord[2] + action_size]
            new_voxel_cord = (voxel_cord[0], voxel_cord[1], voxel_cord[2] + 1)

        else:  # 5
            new_lps_cord = [lps_cord[0], lps_cord[1], lps_cord[2] - action_size]
            new_voxel_cord = (voxel_cord[0], voxel_cord[1], voxel_cord[2] - 1)

        self.agent_location_history_physical[agent_id].append(np.array(new_lps_cord))
        self.agent_location_history_voxel[agent_id].append(new_voxel_cord)

    def __revert_action(self, agent_id):
        # this only gets called when the agent hits the border to maintain a valid internal state

        # keep the invalid step to track oscillations at the border of the image
        self.agent_location_history_physical[agent_id].append(
            self.agent_location_history_physical[agent_id][-2]
        )
        # voxel_list can potentially only have one element
        if len(self.agent_location_history_voxel[agent_id]) > 1:
            self.agent_location_history_voxel[agent_id].append(
                self.agent_location_history_voxel[agent_id][-2]
            )
        else:
            self.agent_location_history_voxel[agent_id].append(
                self.agent_location_history_voxel[agent_id][-1]
            )

    def __get_best_loc(self, agent_id):
        # based on Amir Alansary's implementation
        last_q_values_history = self.q_value_history[agent_id][-4:]
        last_loc_history = self.agent_location_history_physical[agent_id][-4:]

        best_qvalue = max(last_q_values_history)
        best_idx = last_q_values_history.index(best_qvalue)
        best_location = last_loc_history[best_idx]

        # NOTE: only return choice if best q_value is positive -- avoid random jumps due to random actions (rand action q_vals are zero)
        if best_qvalue > 0.0:
            return best_location
        else:
            return self.agent_location_history_physical[agent_id][-1]

    # optionally pass in the episode data you want to run (good for inference and distributed evalutaion)
    def reset(self, episode_data=None):
        if episode_data is None:
            try:
                self.episode_data = next(self.dataset_iter)
            except StopIteration:
                # at the end of the list restart iterator
                self.dataset_iter = iter(self.dataset)
                self.episode_data = next(self.dataset_iter)
        else:
            self.episode_data = episode_data

        # print(self.episode_data["id"])
        if self.mode != "inf":
            self.goal_locations = {
                agent_id: self.episode_data[f"fcsv"][agent_id]
                for agent_id in self.landmark_names
            }
        else:
            self.goal_locations = {
                agent_id: [0, 0, 0] for agent_id in self.landmark_names
            }

        self.location_priors = None
        if self.use_priors:
            if self.image_tag == "og":
                self.location_priors = self.episode_data["prior"]
            else:
                prior_lmks = deepcopy(self.landmark_names)
                for i in ["AC", "PC", "RP"]:
                    prior_lmks.append(i)
                priors_tfm = Compose(
                    [
                        LoadCSVd(keys=["priors"], pd_kwargs=PANDAS_FCSV_KWARGS),
                        ExtractLandmarks2d(keys=["priors"], landmark_names=prior_lmks),
                        TransformLandmarksd(
                            keys=["priors", "first_stage"],
                            landmarks_to_transform=self.landmark_names,
                        ),
                    ]
                )
                priors_file = {
                    "priors": self.priors,
                    "first_stage": self.episode_data["Primary_lmks"],
                }
                self.location_priors = priors_tfm(priors_file)["landmarks"]

        self.resolution_index = {agent_id: 0 for agent_id in self.landmark_names}
        self.step_count = 0

        if self.location_priors is None:
            # nan check
            if self.episode_data["otsu_center"] is not None:
                centroid = np.array(self.episode_data["otsu_center"])
                # those possitions are the averages computed on the training set
                AC_init = centroid + np.array(
                    [0.4058685664054529, -7.651496864631718, -5.30820190739466]
                )
                PC_init = centroid + np.array(
                    [0.43278768522889216, 4.775979407693639, -4.490543106602362]
                )
                RP_init = centroid + np.array(
                    [0.4344664367273539, 7.473449209976261, -18.582815058809526]
                )
                self.agent_location_history_physical = {
                    "AC": [AC_init],
                    "PC": [PC_init],
                    "RP": [RP_init],
                }
            else:
                # backup init -- center of voxel space
                physical_space_center = self.episode_data[
                    "image"
                ].TransformIndexToPhysicalPoint(
                    [
                        int(i)
                        for i in np.array(
                            self.episode_data["image"]
                            .GetLargestPossibleRegion()
                            .GetSize()
                        )
                        / 2
                    ]
                )

                self.agent_location_history_physical = {
                    agent_id: [physical_space_center]
                    for agent_id in self.landmark_names
                }
        else:
            # use priors
            self.agent_location_history_physical = {
                agent_id: [self.location_priors[agent_id]]
                for agent_id in self.landmark_names
            }

        if self.noisy_initialization > 0 and self.mode == "train":
            for agent_id in self.landmark_names:
                noise = (
                    self.noisy_initialization * 2 * np.random.rand(3)
                ) - self.noisy_initialization
                self.agent_location_history_physical[agent_id][0] = (
                    self.agent_location_history_physical[agent_id][0] + noise
                )

        self.agent_location_history_voxel = {
            agent_id: [(0, 0, 0)] for agent_id in self.landmark_names
        }
        self.step_count = 0

        # used to normalize agent location vector
        self.image_diagonal_size = np.linalg.norm(
            np.array(self.episode_data["image"].GetSpacing())
            * np.array(self.episode_data["image"].GetLargestPossibleRegion().GetSize())
        )

        self.obs_history = {}
        self.q_value_history = {agent_id: [] for agent_id in self.landmark_names}

        return {
            agent_id: self.__observation(agent_id) for agent_id in self.landmark_names
        }

    def step(self, action_dict, q_vals):
        # ic(action_dict)
        self.step_count += 1

        observations = {}
        rewards = {}
        dones = {}
        infos = {}

        # save off q values
        for agent_id, q_val in q_vals.items():
            self.q_value_history[agent_id].append(q_val)

        for agent_id, action in action_dict.items():
            self.__apply_action(agent_id, action)

        for agent_id in action_dict.keys():
            # determine if the agent is done
            dones[agent_id] = self.__done(agent_id)
            dones["__all__"] = all(dones.values())
            # if oscillating move to next resolution
            if self.__oscillating(agent_id):
                # print(self.resolution_index[agent_id])
                if self.resolution_index[agent_id] < len(self.resolutions) - 1:
                    # if agent isn't at highest res yet
                    self.resolution_index[agent_id] = (
                        self.resolution_index[agent_id] + 1
                    )  # move to next resolution

                    self.agent_location_history_voxel[agent_id] = [
                        (0, 0, 0)
                    ]  # reset voxel history

                # get best_loc
                self.agent_location_history_physical[agent_id].append(
                    self.__get_best_loc(agent_id)
                )

            # calculate reward
            rewards[agent_id] = self.__reward(agent_id)

            # clip rewards
            rewards[agent_id] = min(
                max(rewards[agent_id], -1 * self.clip_reward), self.clip_reward
            )

            # check inbounds
            if not self.__inbounds(agent_id):
                self.__revert_action(agent_id)
                rewards[agent_id] = -1.0

            # get observations
            observations[agent_id] = self.__observation(agent_id)

            infos[agent_id] = {}
            infos[agent_id][
                "physical_space_location"
            ] = self.agent_location_history_physical[agent_id][-1].tolist()
            infos[agent_id]["euc_dist"] = np.linalg.norm(
                self.agent_location_history_physical[agent_id][-1]
                - self.goal_locations[agent_id]
            )
            infos[agent_id]["res"] = self.resolutions[self.resolution_index[agent_id]]
            infos[agent_id]["steps"] = self.step_count

            if self.mode == "test":
                index_location = np.array(
                    self.episode_data["image"].TransformPhysicalPointToIndex(
                        self.agent_location_history_physical[agent_id][-1]
                    )
                )
                index_goal = np.array(
                    self.episode_data["image"].TransformPhysicalPointToIndex(
                        self.goal_locations[agent_id]
                    )
                )
                infos[agent_id]["vox_euc_dist"] = np.linalg.norm(
                    index_location - index_goal
                )

        return (observations, rewards, dones, infos)

    def seed(self, seed=None):
        random.seed(seed)
