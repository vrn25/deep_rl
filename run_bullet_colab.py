import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pybullet_envs as pe
import imageio

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Bullet environments')
parser.add_argument('--env', type=str, default='HopperBulletEnv-v0', 
                    help='choose an environment between Hopper-v2, HalfCheetah-v2, Ant-v2 and Humanoid-v2')
parser.add_argument('--algo', type=str, default='atac', 
                    help='select an algorithm among vpg, npg, trpo, ppo, ddpg, td3, sac, asac, tac, atac')
parser.add_argument('--phase', type=str, default='train',
                    help='choose between training phase and testing phase')
parser.add_argument('--render', action='store_true', default=False,
                    help='if you want to render, set this to True')
parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the saved model name, and load it')
parser.add_argument('--seed', type=int, default=0, 
                    help='seed for random number generators')
parser.add_argument('--iterations', type=int, default=200, 
                    help='iterations to run and train agent')
parser.add_argument('--steps_per_iter', type=int, default=5000, 
                    help='steps of interaction for the agent and the environment in each epoch')
parser.add_argument('--max_step', type=int, default=1000,
                    help='max episode step')
parser.add_argument('--tensorboard', action='store_true', default=False)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--run_index', type=int, default=0)
parser.add_argument('--drive_location', type=str, default='')
parser.add_argument('--save_freq', type=int, default=500)
parser.add_argument('--eval_episodes', type=int, default=10)

args = parser.parse_args()
if args.gpu_index>=0 and not torch.cuda.is_available():
    raise Exception('GPU not available!')
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

if args.algo == 'vpg':
    from agents.vpg import Agent
elif args.algo == 'npg':
    from agents.trpo import Agent
elif args.algo == 'trpo':
    from agents.trpo import Agent
elif args.algo == 'ppo':
    from agents.ppo import Agent
elif args.algo == 'ddpg':
    from agents.ddpg import Agent
elif args.algo == 'td3':
    from agents.td3 import Agent
elif args.algo == 'sac':
    from agents.sac import Agent
elif args.algo == 'asac': # Automating entropy adjustment on SAC
    from agents.sac import Agent
elif args.algo == 'tac': 
    from agents.sac import Agent
elif args.algo == 'atac': # Automating entropy adjustment on TAC
    from agents.sac import Agent


def main():
    """Main."""
    # Initialize environment
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_limits = [env.observation_space.low[0], env.observation_space.high[0]]
    act_limits = [env.action_space.low[0], env.action_space.high[0]]
    act_limit = act_limits[1]

    print('---------------------------------------')
    print('Environment:', args.env)
    print('Algorithm:', args.algo)
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)
    print('State limit:', obs_limits)
    print('Action limit:', act_limits)
    print('---------------------------------------')

    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create an agent
    if args.algo == 'ddpg' or args.algo == 'td3':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      expl_before=10000, 
                      act_noise=0.1, 
                      hidden_sizes=(256,256), 
                      buffer_size=int(1e6), 
                      batch_size=256,
                      policy_lr=3e-4, 
                      qf_lr=3e-4)
    elif args.algo == 'sac':                                                                                    
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit,                   
                      expl_before=10000,                                
                      alpha=0.2,                        # In HalfCheetah-v2 and Ant-v2, SAC with 0.2  
                      hidden_sizes=(256,256),           # shows the best performance in entropy coefficient 
                      buffer_size=int(1e6),             # while, in Humanoid-v2, SAC with 0.05 shows the best performance.
                      batch_size=256,
                      policy_lr=3e-4, 
                      qf_lr=3e-4)     
    elif args.algo == 'asac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      expl_before=10000, 
                      automatic_entropy_tuning=True, 
                      hidden_sizes=(256,256), 
                      buffer_size=int(1e6), 
                      batch_size=256,
                      policy_lr=3e-4,
                      qf_lr=3e-4)
    elif args.algo == 'tac':                                                                                    
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit,                   
                      expl_before=10000,                                
                      alpha=0.2,                                       
                      log_type='log-q',                 
                      entropic_index=1.2,               
                      hidden_sizes=(256,256),          
                      buffer_size=int(1e6), 
                      batch_size=256,
                      policy_lr=3e-4,
                      qf_lr=3e-4)
    elif args.algo == 'atac':
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, 
                      expl_before=10000, 
                      log_type='log-q', 
                      entropic_index=1.2, 
                      automatic_entropy_tuning=True,
                      hidden_sizes=(256,256), 
                      buffer_size=int(1e6), 
                      batch_size=256,
                      policy_lr=3e-4,
                      qf_lr=3e-4)
    else: # vpg, npg, trpo, ppo
        agent = Agent(env, args, device, obs_dim, act_dim, act_limit, sample_size=4096)

    # If we have a saved model, load it
    if args.load is not '':
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path, map_location=device)
        agent.policy.load_state_dict(pretrained_model)

    # Create a SummaryWriter object by TensorBoard
    if args.tensorboard:# and args.load == '':
        dir_name = 'runs/' + args.env + '/' \
                           + args.algo + '/run_' + str(args.run_index) \
                           + '_seed_' + str(args.seed)
                           #+ '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir=dir_name)

    start_time = time.time()

    # total number of interactions with env till now
    total_num_steps = 0
    # total sum of rewards till now
    train_sum_returns = 0.
    train_num_episodes = 0
    
    train_average_return_list = []
    train_each_episode_return_list = []
    eval_average_returns_per_itr_list = []

    if args.phase == 'test':
        images = []

    # Main loop
    for i in range(args.iterations):
        # Perform the training phase, during which the agent learns
        # one iteration corresponds to approximately "args.steps_per_itr" number of interactions with the env
        if args.phase == 'train':
            train_step_count = 0

            while train_step_count <= args.steps_per_iter:
                agent.eval_mode = False
                
                # Run one episode
                # train_step_length is the number of steps in the episode
                train_step_length, train_episode_return, _ = agent.run(args.max_step)
                
                total_num_steps += train_step_length
                train_step_count += train_step_length
                train_sum_returns += train_episode_return
                train_each_episode_return_list.append([train_episode_return, total_num_steps])
                train_num_episodes += 1

                train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0
                train_average_return_list.append([train_average_return, total_num_steps])

                # Log experiment result for training steps
                if args.tensorboard:# and args.load is None:
                    # (total sum of rewards till now)/number of episodes: per episode return
                    writer.add_scalar('Train/AverageReturns (sum_rews_till_now/num_eps_till_now)', train_average_return, total_num_steps)
                    # total sum of rewards in this episode
                    writer.add_scalar('Train/EachEpisodeReturns', train_episode_return, total_num_steps)
                    if args.algo == 'asac' or args.algo == 'atac':
                        writer.add_scalar('Train/Alpha', agent.alpha, total_num_steps)

        # Perform the evaluation phase -- no learning
        eval_sum_returns = 0.
        eval_num_episodes = 0
        agent.eval_mode = True

        for _ in range(args.eval_episodes):
            # Run one episode
            eval_step_length, eval_episode_return, imgs = agent.run(args.max_step)

            eval_sum_returns += eval_episode_return
            eval_num_episodes += 1
            if args.phase == 'test':
                images += imgs

        eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

        # Log experiment result for evaluation steps
        if args.tensorboard:# and args.load is None:
            writer.add_scalar('Eval/AverageReturns', eval_average_return, total_num_steps)
            writer.add_scalar('Eval/AverageReturnsPerIteration', eval_average_return, i)
            #writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, total_num_steps)
        eval_average_returns_per_itr_list.append([eval_average_return, total_num_steps])

        if args.phase == 'train':
            print('---------------------------------------')
            print('Iterations:', i + 1)
            print('Steps (interactions with env till now):', total_num_steps)
            print('Episodes (till now):', train_num_episodes)
            print('EpisodeReturn (return in the latest eps):', round(train_episode_return, 2))
            print('AverageReturn (total reward till now/num_eps):', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            #print('EvalEpisodeReturn:', round(eval_episode_return, 2))
            print('EvalAverageReturn (EAR):', round(eval_average_return, 2))
            print('OtherLogs:', agent.logger)
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            # Save the trained model
            if (i + 1) % args.save_freq == 0 or i==0 or i==args.iterations-1:
                save_path = args.drive_location + '/' + args.env + '/' + args.algo + '/'
                np_file = save_path + args.env + '_' + args.algo + '_' + 'np_arrays_run_' + str(args.run_index) + '_s_' + str(args.seed) + '.npz'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                
                ckpt_path = os.path.join(save_path + args.env + '_' + args.algo + '_' + 'policy_run_' + str(args.run_index) \
                                                                    + '_seed_' + str(args.seed) \
                                                                    + '_itr_' + str(i + 1) \
                                                                    + '_ear_' + str(round(eval_average_return, 2)) + '.pt')
                
                torch.save(agent.policy.state_dict(), ckpt_path)
                with open(np_file, 'wb') as np_f:
                    np.savez(np_f, eval_average_returns_per_itr_list=np.array(eval_average_returns_per_itr_list), \
                    train_average_return_list=np.array(train_average_return_list), \
                    train_each_episode_return_list=np.array(train_each_episode_return_list), \
                    train_num_episodes=np.array(train_num_episodes))
        elif args.phase == 'test':
            print('---------------------------------------')
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalEpisodeReturn:', round(eval_episode_return, 2))
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

    if args.phase == 'test':
        imageio.mimsave('video.mp4', [np.array(img) for i, img in enumerate(images)], fps=40)
            
if __name__ == "__main__":
    main()
