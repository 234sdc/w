clc
clear all
close all

% 设置随机种子，便于实验复现
RANDOM_SEED = 42;  % 可以修改这个值来改变随机性
rng(RANDOM_SEED, 'twister');  % 使用Mersenne Twister算法
rand('seed', RANDOM_SEED);    % 设置rand的种子
randn('seed', RANDOM_SEED);   % 设置randn的种子

% 打印当前种子值，便于记录
fprintf('当前随机种子: %d\n', RANDOM_SEED);

N = 20;               % 训练轮数
ini_health = 20;      % 初始健康值
nums = 10;            % 任务数量
thre = 1000;          % 每轮的总时间步数
dur = 500;            % 训练频率
counter = 1;

% 初始化经验回放缓冲区
record_input = zeros(dur, 62);    % 状态
record_nextstate = zeros(dur, 62); % 下一状态
record_action = zeros(dur, 1);     % 动作（索引）
record_reward = zeros(dur, 1);     % 奖励
record_terminal = zeros(dur, 1);   % 终止状态标志

% DQN超参数
epsilon = 0.9;        % 初始探索率
epsilon_min = 0.1;    % 最小探索率
epsilon_decay = 0.995;% 探索率衰减
gamma = 0.95;         % 折扣因子
learning_rate = 0.001;% 学习率
minibatch_size = 32;  % 小批量大小
target_update_freq = 10; % 目标网络更新频率

% 创建DQN网络
% 主网络
net = [];
net.layers = [];
net.layers{1} = struct('type', 'input', 'size', [62 1]);
net.layers{2} = struct('type', 'fc', 'size', [128 62], 'activation', 'relu');
net.layers{3} = struct('type', 'fc', 'size', [64 128], 'activation', 'relu');
net.layers{4} = struct('type', 'fc', 'size', [3 64], 'activation', 'linear');

% 初始化网络权重
for i = 2:length(net.layers)
    if strcmp(net.layers{i}.type, 'fc')
        % Xavier初始化
        input_size = net.layers{i}.size(2);
        output_size = net.layers{i}.size(1);
        net.layers{i}.weights = randn(output_size, input_size) * sqrt(2/(input_size + output_size));
        net.layers{i}.bias = zeros(output_size, 1);
    end
end

% 创建目标网络（主网络的副本）
target_net = net;

% 前向传播函数
forward = @(network, x) forward_pass(network, x);

% 初始化分数跟踪
score = zeros(1, N);
update_counter = 0;

for episode = 1:N
    t = 1;
    
    % 初始化环境
    states = mission_profile(nums);
    health_true = ini_health;
    health_sense = ini_health + ini_health/10*randn;
    input_set = [states.length  states.mission_timepoint  states.mission_strengh  ...
            states.mission_reward  states.maintenance_time  states.maintenance_cost];
    state = [t health_sense reshape(input_set', 1, [])];
    
    reward_total = 0;
    cost_total = 0;
    
    while t < thre
        % Epsilon-贪婪动作选择
        if rand <= epsilon
            action_idx = randi(3);
        else
            % 通过网络进行前向传播
            q_values = forward(net, state');
            [~, action_idx] = max(q_values);
        end
        
        % 创建独热编码动作
        action = zeros(3, 1);
        action(action_idx) = 1;
        
        % 执行动作并观察新状态和奖励
        [reward_temp, cost_temp, next_state] = reward_cal(state, action, ini_health);
        
        % 检查是否为终止状态
        terminal = (t >= thre);
        
        % 将经验存储在回放缓冲区中
        if counter > dur
            counter = 1;
        end
        record_input(counter, :) = state;
        record_nextstate(counter, :) = next_state;
        record_action(counter, :) = action_idx;
        record_reward(counter, :) = reward_temp - cost_temp;
        record_terminal(counter, :) = terminal;
        counter = counter + 1;
        
        % 更新总奖励和总成本
        reward_total = reward_total + reward_temp;
        cost_total = cost_total + cost_temp;
        
        % 经验回放和网络更新
        if mod(counter, minibatch_size) == 0
            % 从回放缓冲区中随机采样小批量
            indices = randi(min(counter-1, dur), 1, minibatch_size);
            
            % 提取批量数据
            state_batch = record_input(indices, :);
            next_state_batch = record_nextstate(indices, :);
            action_batch = record_action(indices, :);
            reward_batch = record_reward(indices, :);
            terminal_batch = record_terminal(indices, :);
            
            % 计算目标Q值
            target_q = zeros(minibatch_size, 3);
            
            for i = 1:minibatch_size
                % 获取当前状态的Q值
                target_q(i, :) = forward(net, state_batch(i, :)');
                
                % 使用目标网络获取下一状态的最大Q值
                next_q = forward(target_net, next_state_batch(i, :)');
                [max_q, ~] = max(next_q);
                
                % 更新所选动作的目标
                if terminal_batch(i)
                    target_q(i, action_batch(i)) = reward_batch(i);
                else
                    target_q(i, action_batch(i)) = reward_batch(i) + gamma * max_q;
                end
            end
            
            % 更新网络权重
            net = update_network(net, state_batch, target_q, learning_rate);
            
            % 更新目标网络
            update_counter = update_counter + 1;
            if mod(update_counter, target_update_freq) == 0
                target_net = net;
            end
        end
        
        % 更新状态和时间
        state = next_state;
        t = state(1);
        
        % 衰减探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay);
    end
    
    % 记录本轮的分数
    score(episode) = reward_total - cost_total;
    
    % 打印进度
    fprintf('轮次 %d/%d, 分数: %.2f, 探索率: %.2f\n', episode, N, score(episode), epsilon);
end

% 绘制结果
figure
subplot(211)
plot(score);
title('每轮分数');
xlabel('轮次');
ylabel('分数');

subplot(212)
hist(score, 30);
title('分数分布');
xlabel('分数');
ylabel('频率');

fprintf('平均分数: %.2f\n', mean(score));
fprintf('分数标准差: %.2f\n', std(score));

%% 辅助函数

function y = forward_pass(network, x)
    % 输入层
    a = x;
    
    % 通过每一层进行前向传播
    for i = 2:length(network.layers)
        if strcmp(network.layers{i}.type, 'fc')
            % 全连接层
            z = network.layers{i}.weights * a + network.layers{i}.bias;
            
            % 应用激活函数
            if strcmp(network.layers{i}.activation, 'relu')
                a = max(0, z);
            elseif strcmp(network.layers{i}.activation, 'linear')
                a = z;
            end
        end
    end
    
    % 输出
    y = a;
end

function network = update_network(network, states, targets, learning_rate)
    % 反向传播的简单实现
    for i = 1:size(states, 1)
        % 前向传播
        activations = cell(1, length(network.layers));
        activations{1} = states(i, :)';
        
        for j = 2:length(network.layers)
            if strcmp(network.layers{j}.type, 'fc')
                z = network.layers{j}.weights * activations{j-1} + network.layers{j}.bias;
                
                if strcmp(network.layers{j}.activation, 'relu')
                    activations{j} = max(0, z);
                elseif strcmp(network.layers{j}.activation, 'linear')
                    activations{j} = z;
                end
            end
        end
        
        % 反向传播
        % 计算输出层的误差
        error = targets(i, :)' - activations{end};
        
        % 反向传播误差
        for j = length(network.layers):-1:2
            if strcmp(network.layers{j}.type, 'fc')
                % 计算梯度
                if strcmp(network.layers{j}.activation, 'relu')
                    delta = error .* (activations{j} > 0);
                elseif strcmp(network.layers{j}.activation, 'linear')
                    delta = error;
                end
                
                % 更新权重和偏置
                network.layers{j}.weights = network.layers{j}.weights + learning_rate * delta * activations{j-1}';
                network.layers{j}.bias = network.layers{j}.bias + learning_rate * delta;
                
                % 将误差传播到前一层
                if j > 2
                    error = network.layers{j}.weights' * delta;
                end
            end
        end
    end
end