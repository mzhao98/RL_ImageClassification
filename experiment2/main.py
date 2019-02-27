


def custom_replace(tensor, on_zero, on_non_zero):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

class Plot(object):
    def __init__(self, title, port=8080):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel):
        win = self.viz.scatter(
            X=numpy.zeros((1, 2)),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_scatterplot(self, name, x, y):
        self.viz.updateTrace(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.windows[name]
        )


def main():
    import copy
    import glob
    import os
    import time
    import matplotlib.pyplot as plt

    import gym
    import numpy as np
    import torch
    torch.multiprocessing.set_start_method('spawn')

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from gym.spaces import Discrete

    from arguments import get_args
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from envs import make_env
    from img_env import ImgEnv, IMG_ENVS
    from model import Policy
    from storage import RolloutStorage
    from utils import update_current_obs, agent1_eval_episode, agent2_eval_episode
    from torchvision import transforms
    from visdom import Visdom

    import algo

    # viz = Visdom(port=8097)

    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    plot_rewards = []
    plot_policy_loss = []
    plot_value_loss = []
    # x = np.array([0])
    # y = np.array([0])
    # counter = 0
    # win = viz.line(
    #     X=x,
    #     Y=y,
    #     win="test1",
    #     name='Line1',
    #     opts=dict(
    #         title='Reward',
    #     )
    #     )
    # win2 = viz.line(
    #     X=x,
    #     Y=y,
    #     win="test2",
    #     name='Line2',
    #     opts=dict(
    #         title='Policy Loss',
    #     )
    #     )
    # win3 = viz.line(
    #     X=x,
    #     Y=y,
    #     win="test3",
    #     name='Line3',
    #     opts=dict(
    #         title='Value Loss',
    #     )
    #     )

    args = get_args()
    if args.no_cuda:
        args.cuda = False
    print(args)
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    toprint = ['seed', 'lr', 'nat', 'resnet']
    if args.env_name in IMG_ENVS:
        toprint += ['window', 'max_steps']
    toprint.sort()
    name = args.tag
    args_param = vars(args)
    os.makedirs(os.path.join(args.out_dir, args.env_name), exist_ok=True)
    for arg in toprint:
        if arg in args_param and (args_param[arg] or arg in ['gamma', 'seed']):
            if args_param[arg] is True:
                name += '{}_'.format(arg)
            else:
                name += '{}{}_'.format(arg, args_param[arg])
    model_dir = os.path.join(args.out_dir, args.env_name, args.algo)
    os.makedirs(model_dir, exist_ok=True)

    results_dict = {
        'episodes': [],
        'rewards': [],
        'args': args
    }
    torch.set_num_threads(1)
    eval_env = make_env(args, 'cifar10', args.seed, 1, None,
            args.add_timestep, natural=args.nat, train=False)
    envs = make_env(args, 'cifar10', args.seed, 1, None,
            args.add_timestep, natural=args.nat, train=True)
                
    #print(envs)
    # envs = envs[0]
    

    # if args.num_processes > 1:
    #     envs = SubprocVecEnv(envs)
    # else:
    #     envs = DummyVecEnv(envs)
    # eval_env = DummyVecEnv(eval_env)
    # if len(envs.observation_space.shape) == 1:
    #     envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic1 = Policy(obs_shape, envs.action_space, args.recurrent_policy,
                          dataset=args.env_name, resnet=args.resnet,
                          pretrained=args.pretrained)

    actor_critic2 = Policy(obs_shape, envs.action_space, args.recurrent_policy,
                          dataset=args.env_name, resnet=args.resnet,
                          pretrained=args.pretrained)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic1.cuda()
        actor_critic2.cuda()

    if args.algo == 'a2c':
        agent1 = algo.A2C_ACKTR(actor_critic1, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
        agent2 = algo.A2C_ACKTR(actor_critic2, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent1 = algo.PPO(actor_critic1, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
        agent2 = algo.PPO(actor_critic2, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent1 = algo.A2C_ACKTR(actor_critic1, args.value_loss_coef,
                               args.entropy_coef, acktr=True)
        agent2 = algo.A2C_ACKTR(actor_critic2, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    action_space = envs.action_space
    if args.env_name in IMG_ENVS:
        action_space = np.zeros(2)
    # obs_shape = envs.observation_space.shape
    agent1_rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, action_space, actor_critic1.state_size)
    agent1_current_obs = torch.zeros(args.num_processes, *obs_shape)

    agent1_obs = envs.agent1_reset()
    update_current_obs(agent1_obs, agent1_current_obs, obs_shape, args.num_stack)
    agent1_rollouts.observations[0].copy_(agent1_current_obs)

    agent2_rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, action_space, actor_critic2.state_size)
    agent2_current_obs = torch.zeros(args.num_processes, *obs_shape)

    agent2_obs = envs.agent2_reset()
    update_current_obs(agent2_obs, agent2_current_obs, obs_shape, args.num_stack)
    agent2_rollouts.observations[0].copy_(agent2_current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        # envs.display_original(j)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value1, action1, action_log_prob1, states1 = actor_critic1.act(
                        agent1_rollouts.observations[step],
                        agent1_rollouts.states[step],
                        agent1_rollouts.masks[step])
                value2, action2, action_log_prob2, states2 = actor_critic2.act(
                        agent2_rollouts.observations[step],
                        agent2_rollouts.states[step],
                        agent2_rollouts.masks[step])

            cpu_actions1 = action1.squeeze(1).cpu().numpy()
            cpu_actions2 = action2.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs1, reward1, done1, info1 = envs.agent1_step(cpu_actions1)
            obs2, reward2, done2, info2 = envs.agent2_step(cpu_actions2)

            # SIMPLE HEURISTIC 1
            # If either agent gets it correct, they are done.
    
            if done1 == True or done2 == True:
                done1 = True
                done2 = True
                done = True
            else:
                done = False



            # envs.display_step(step, j)

            # print("OBS", obs)

            # print("REWARD", reward)
            # print("DONE", done)
            # print("INFO", info)


            reward1 = torch.from_numpy(np.expand_dims(np.stack([reward1]), 1)).float()
            reward2 = torch.from_numpy(np.expand_dims(np.stack([reward2]), 1)).float()
            reward = (reward1+reward2)
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in [done]])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if agent1_current_obs.dim() == 4:
                agent1_current_obs *= masks.unsqueeze(2).unsqueeze(2)
                agent2_current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                agent1_current_obs *= masks
                agent2_current_obs *= masks


            update_current_obs(agent1_obs, agent1_current_obs, obs_shape, args.num_stack)
            agent1_rollouts.insert(agent1_current_obs, states1, action1, action_log_prob1, value1, reward, masks)

            update_current_obs(agent2_obs, agent2_current_obs, obs_shape, args.num_stack)
            agent2_rollouts.insert(agent2_current_obs, states2, action2, action_log_prob2, value2, reward, masks)

            # print("envs.curr_img SHAPE: ", envs.curr_img.shape)
            #display_state = envs.curr_img
            # display_state[:, envs.pos[0]:envs.pos[0]+envs.window, envs.pos[1]:envs.pos[1]+envs.window] = 5
            # display_state = custom_replace(display_state, 1, 0)
            # display_state[:, envs.pos[0]:envs.pos[0]+envs.window, envs.pos[1]:envs.pos[1]+envs.window] = \
            #     envs.curr_img[:, envs.pos[0]:envs.pos[0]+envs.window, envs.pos[1]:envs.pos[1]+envs.window]
            # img = transforms.ToPILImage()(display_state)
            # img.save("state_cifar/"+"state"+str(j)+"_"+str(step)+".png")

        with torch.no_grad():
            next_value1 = actor_critic1.get_value(agent1_rollouts.observations[-1],
                                                agent1_rollouts.states[-1],
                                                agent1_rollouts.masks[-1]).detach()
            next_value2 = actor_critic2.get_value(agent2_rollouts.observations[-1],
                                                agent2_rollouts.states[-1],
                                                agent2_rollouts.masks[-1]).detach()

        agent1_rollouts.compute_returns(next_value1, args.use_gae, args.gamma, args.tau)
        value_loss1, action_loss1, dist_entropy1 = agent1.update(agent1_rollouts)
        agent1_rollouts.after_update()

        agent2_rollouts.compute_returns(next_value2, args.use_gae, args.gamma, args.tau)
        value_loss2, action_loss2, dist_entropy2 = agent2.update(agent2_rollouts)
        agent2_rollouts.after_update()

        

        if j % args.save_interval == 0:
            torch.save((actor_critic1.state_dict(), results_dict), os.path.join(
                model_dir, name + 'cifar_model_ppo_ex2_agent1.pt'))
            torch.save((actor_critic2.state_dict(), results_dict), os.path.join(
                model_dir, name + 'cifar_model_ppo_ex2_agent2.pt'))

        if j % args.log_interval == 0:
            end = time.time()
            total_reward1 = agent1_eval_episode(eval_env, actor_critic1, args)
            total_reward2 = agent2_eval_episode(eval_env, actor_critic2, args)

            total_reward = (total_reward1+total_reward2)
            value_loss = (value_loss1+value_loss2)
            action_loss = (action_loss1+action_loss2)
            dist_entropy = (dist_entropy1+dist_entropy2)

            results_dict['rewards'].append(total_reward)
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, reward {:.1f} entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       np.mean(results_dict['rewards'][-10:]), dist_entropy,
                       value_loss, action_loss))


            plot_rewards.append(np.mean(results_dict['rewards'][-10:]))
            plot_policy_loss.append(action_loss)
            plot_value_loss.append(value_loss)


    plt.plot(range(len(plot_rewards)), plot_rewards)
    plt.savefig("rewards_multi_1.png")
    plt.close()

    
    plt.plot(range(len(plot_policy_loss)), plot_policy_loss)
    plt.savefig("policyloss_multi_1.png")
    plt.close()

    
    plt.plot(range(len(plot_value_loss)), plot_value_loss)
    plt.savefig("valueloss_multi_1.png")
    plt.close()



if __name__ == "__main__":
    main()
