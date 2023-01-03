from datetime import datetime
import itk
import numpy as np
import torch

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PANDAS_FCSV_KWARGS = {
    "comment": "#",
    "names": [
        "id",
        "x",
        "y",
        "z",
        "ow",
        "ox",
        "oy",
        "oz",
        "vis",
        "sel",
        "lock",
        "label",
        "desc",
        "associatedNodeID",
    ],
}


def policy_mapping_fn(x):
    return x


def observations_to_tensor(obs, landmark_names, state_size):
    spatial_state_tensor = torch.zeros(
        (1, len(landmark_names), len(obs[list(obs.keys())[0]]), *state_size)
    )
    for idx, agent_id in enumerate(landmark_names):
        if agent_id in obs.keys():
            spatial_state_tensor[0][idx] = torch.Tensor(obs[agent_id])

    output = spatial_state_tensor

    return output


# if you pass in the trainer this will be a training episode
def run_episode(env, policy_network, eps, episode_data=None):
    if episode_data is None:
        obs = env.reset()
    else:
        obs = env.reset(episode_data=episode_data)

    dones = {agent_id: False for agent_id in obs.keys()}
    dones["__all__"] = False

    cumulative_rewards = {agent_id: 0 for agent_id in obs.keys()}

    final_infos = {}
    losses = []
    episode_step = 0

    while not dones["__all__"]:
        actions = {}
        q_vals = {}

        # get next action
        #################

        # introduce randomness to the search
        if np.random.random() < eps:
            # random action
            for agent_id in obs.keys():
                if not dones[agent_id]:
                    actions[agent_id] = env.action_space.sample()
                    # no q val for random actions
                    q_vals[agent_id] = 0.0

        else:
            # network action
            network_input = observations_to_tensor(
                obs,
                env.landmark_names,
                env.spatial_state_size,
            )
            with torch.no_grad():
                action_tensor = policy_network(network_input)

            tmp_max = torch.max(action_tensor, dim=2)
            q_val_arr = tmp_max[0].squeeze(0)
            optimal_action_list = tmp_max[1].squeeze(0)

            for agent_idx, agent_id in enumerate(env.landmark_names):
                if agent_id in obs.keys() and not dones[agent_id]:
                    actions[agent_id] = int(optimal_action_list[agent_idx])
                    q_vals[agent_id] = float(q_val_arr[agent_idx])
        #####################

        # appply the action to env state
        next_obs, rewards, dones, infos = env.step(actions, q_vals)

        # storing metrics -- euclidean distance error and reward
        for agent_id in next_obs.keys():
            if dones[agent_id]:
                final_infos[agent_id] = infos[agent_id]

            cumulative_rewards[agent_id] = (
                cumulative_rewards[agent_id] + rewards[agent_id]
            )

        obs = next_obs
        episode_step = episode_step + 1

    return cumulative_rewards, final_infos, losses


def extract_state_from_image(image, state_size, agent_location, resolution):

    IMAGE_TYPE = itk.Image[itk.UC, 3]
    IDENTITY_TRANSFORM = itk.IdentityTransform[itk.D, 3].New()
    IDENTITY_DIRECTION = IMAGE_TYPE.New().GetDirection()
    IDENTITY_DIRECTION.SetIdentity()
    LINEAR_INTERPOLATOR = itk.LinearInterpolateImageFunction[IMAGE_TYPE, itk.D].New()

    resampler = itk.ResampleImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
    resampler.SetOutputDirection(IDENTITY_DIRECTION)
    resampler.SetSize(state_size)
    resampler.SetInterpolator(LINEAR_INTERPOLATOR)
    resampler.SetTransform(IDENTITY_TRANSFORM)

    physical_space_origin_offset = resolution * (np.array(state_size) / 2)
    output_origin = agent_location - physical_space_origin_offset

    resampler.SetOutputOrigin(output_origin)
    resampler.SetOutputSpacing([resolution] * 3)
    resampler.SetInput(image)

    resampler.UpdateLargestPossibleRegion()
    output = resampler.GetOutput()

    return itk.array_from_image(output)


def cartesian2spherical(cart_cord):
    rho = np.linalg.norm(cart_cord)
    theta = np.arctan(cart_cord[1] / cart_cord[0])
    phi = np.arccos(cart_cord[2] / rho)
    return rho, theta if theta is not np.nan else 0.0, phi if phi is not np.nan else 0.0
